"""
Stack Applications and Algorithms
=================================

Topics: Advanced stack algorithms, histogram problems, monotonic stack
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix
Difficulty: Medium to Hard
Time Complexity: Often O(n) with stack optimization
Space Complexity: O(n) for stack storage
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import deque

class StackAlgorithms:
    
    def __init__(self):
        """Initialize with solution tracking"""
        self.solution_steps = []
    
    # ==========================================
    # 1. MONOTONIC STACK PROBLEMS
    # ==========================================
    
    def next_greater_elements(self, nums: List[int]) -> List[int]:
        """
        Find next greater element for each element using monotonic stack
        
        Company: Amazon, Microsoft, Google
        Difficulty: Medium
        Time: O(n), Space: O(n)
        
        Example: [2,1,2,4,3,1] → [4,2,4,-1,4,4]
        """
        result = [-1] * len(nums)
        stack = []  # Store indices
        
        print(f"Finding next greater elements for: {nums}")
        print("Using monotonic decreasing stack (stores indices)")
        print()
        
        for i in range(len(nums)):
            print(f"Step {i+1}: Processing nums[{i}] = {nums[i]}")
            
            # Pop elements smaller than current element
            while stack and nums[stack[-1]] < nums[i]:
                index = stack.pop()
                result[index] = nums[i]
                print(f"   Found next greater for nums[{index}] = {nums[index]}: {nums[i]}")
                print(f"   Stack after pop: {[nums[idx] for idx in stack] if stack else 'empty'}")
            
            # Push current index
            stack.append(i)
            print(f"   Push index {i} (value {nums[i]}) to stack")
            print(f"   Stack: {[nums[idx] for idx in stack]}")
            print(f"   Current result: {result}")
            print()
        
        print(f"Final result: {result}")
        print("Elements still in stack have no next greater element")
        return result
    
    def previous_smaller_elements(self, nums: List[int]) -> List[int]:
        """
        Find previous smaller element for each element
        
        Time: O(n), Space: O(n)
        """
        result = [-1] * len(nums)
        stack = []  # Store indices
        
        print(f"Finding previous smaller elements for: {nums}")
        print("Using monotonic increasing stack")
        print()
        
        for i in range(len(nums)):
            print(f"Step {i+1}: Processing nums[{i}] = {nums[i]}")
            
            # Pop elements greater than or equal to current element
            while stack and nums[stack[-1]] >= nums[i]:
                popped_idx = stack.pop()
                print(f"   Pop nums[{popped_idx}] = {nums[popped_idx]} (>= current)")
            
            # If stack is not empty, top element is previous smaller
            if stack:
                result[i] = nums[stack[-1]]
                print(f"   Previous smaller for nums[{i}] = {nums[i]}: {nums[stack[-1]]}")
            else:
                print(f"   No previous smaller element for nums[{i}] = {nums[i]}")
            
            # Push current index
            stack.append(i)
            print(f"   Stack: {[nums[idx] for idx in stack]}")
            print()
        
        print(f"Final result: {result}")
        return result
    
    def daily_temperatures(self, temperatures: List[int]) -> List[int]:
        """
        Find how many days you have to wait for warmer temperature
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(n), Space: O(n)
        
        Example: [73,74,75,71,69,72,76,73] → [1,1,4,2,1,1,0,0]
        """
        result = [0] * len(temperatures)
        stack = []  # Store indices
        
        print(f"Finding daily temperatures for: {temperatures}")
        print("Result will show days to wait for warmer weather")
        print()
        
        for i in range(len(temperatures)):
            print(f"Day {i}: Temperature = {temperatures[i]}°")
            
            # Pop days with lower temperatures
            while stack and temperatures[stack[-1]] < temperatures[i]:
                prev_day = stack.pop()
                days_wait = i - prev_day
                result[prev_day] = days_wait
                print(f"   Day {prev_day} (temp {temperatures[prev_day]}°) waits {days_wait} days")
                print(f"   Stack after pop: {stack}")
            
            # Push current day
            stack.append(i)
            print(f"   Push day {i} to stack: {stack}")
            print()
        
        print(f"Final result: {result}")
        print("0 means no warmer day found")
        return result
    
    # ==========================================
    # 2. HISTOGRAM PROBLEMS
    # ==========================================
    
    def largest_rectangle_in_histogram(self, heights: List[int]) -> int:
        """
        Find largest rectangle area in histogram using stack
        
        Company: Google, Amazon, Microsoft
        Difficulty: Hard
        Time: O(n), Space: O(n)
        
        Algorithm:
        1. Use stack to store indices of bars
        2. Maintain increasing order in stack
        3. When smaller bar found, calculate area with popped bars
        """
        stack = []
        max_area = 0
        index = 0
        
        print(f"Finding largest rectangle in histogram: {heights}")
        print("Using stack to maintain increasing heights")
        print()
        
        while index < len(heights):
            print(f"Step {index + 1}: Processing height[{index}] = {heights[index]}")
            
            # If current bar is higher, push index to stack
            if not stack or heights[index] >= heights[stack[-1]]:
                stack.append(index)
                print(f"   Height is increasing, push index {index}")
                print(f"   Stack: {stack}")
                index += 1
            else:
                # Pop the top and calculate area
                top_index = stack.pop()
                height = heights[top_index]
                
                # Width calculation
                width = index if not stack else index - stack[-1] - 1
                area = height * width
                
                print(f"   Height decreases, pop index {top_index} (height {height})")
                print(f"   Width = {width}, Area = {height} × {width} = {area}")
                
                max_area = max(max_area, area)
                print(f"   Max area so far: {max_area}")
                print(f"   Stack after pop: {stack}")
        
        # Pop remaining bars and calculate area
        print(f"\nProcessing remaining bars in stack:")
        while stack:
            top_index = stack.pop()
            height = heights[top_index]
            width = index if not stack else index - stack[-1] - 1
            area = height * width
            
            print(f"   Pop index {top_index} (height {height})")
            print(f"   Width = {width}, Area = {height} × {width} = {area}")
            
            max_area = max(max_area, area)
            print(f"   Max area so far: {max_area}")
        
        print(f"\nLargest rectangle area: {max_area}")
        return max_area
    
    def maximal_rectangle_binary_matrix(self, matrix: List[List[str]]) -> int:
        """
        Find largest rectangle of 1s in binary matrix
        
        Company: Google, Amazon
        Difficulty: Hard
        Time: O(m*n), Space: O(n)
        
        Algorithm: Convert to histogram problem for each row
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        print(f"Finding maximal rectangle in binary matrix:")
        for row in matrix:
            print(f"   {row}")
        print()
        
        for i in range(rows):
            print(f"Processing row {i}: {matrix[i]}")
            
            # Update heights array
            for j in range(cols):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            print(f"   Heights array: {heights}")
            
            # Find largest rectangle in current histogram
            area = self.largest_rectangle_in_histogram_simple(heights)
            max_area = max(max_area, area)
            
            print(f"   Largest rectangle in this histogram: {area}")
            print(f"   Max area so far: {max_area}")
            print()
        
        print(f"Final maximal rectangle area: {max_area}")
        return max_area
    
    def largest_rectangle_in_histogram_simple(self, heights: List[int]) -> int:
        """Simplified version for internal use"""
        stack = []
        max_area = 0
        index = 0
        
        while index < len(heights):
            if not stack or heights[index] >= heights[stack[-1]]:
                stack.append(index)
                index += 1
            else:
                top = stack.pop()
                height = heights[top]
                width = index if not stack else index - stack[-1] - 1
                max_area = max(max_area, height * width)
        
        while stack:
            top = stack.pop()
            height = heights[top]
            width = index if not stack else index - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area
    
    # ==========================================
    # 3. PARENTHESES AND BRACKET PROBLEMS
    # ==========================================
    
    def longest_valid_parentheses(self, s: str) -> int:
        """
        Find length of longest valid parentheses substring
        
        Company: Amazon, Google
        Difficulty: Hard
        Time: O(n), Space: O(n)
        
        Example: "(()" → 2, ")()())" → 4
        """
        stack = [-1]  # Base for length calculation
        max_length = 0
        
        print(f"Finding longest valid parentheses in: '{s}'")
        print("Stack stores indices, -1 as base")
        print()
        
        for i, char in enumerate(s):
            print(f"Step {i+1}: Processing '{char}' at index {i}")
            
            if char == '(':
                stack.append(i)
                print(f"   Opening parenthesis: Push index {i}")
                print(f"   Stack: {stack}")
            else:  # char == ')'
                stack.pop()
                print(f"   Closing parenthesis: Pop from stack")
                
                if not stack:
                    # No matching opening parenthesis
                    stack.append(i)
                    print(f"   No matching '(', push index {i} as new base")
                    print(f"   Stack: {stack}")
                else:
                    # Valid parentheses found
                    length = i - stack[-1]
                    max_length = max(max_length, length)
                    print(f"   Valid parentheses found")
                    print(f"   Length = {i} - {stack[-1]} = {length}")
                    print(f"   Max length so far: {max_length}")
                    print(f"   Stack: {stack}")
            print()
        
        print(f"Longest valid parentheses length: {max_length}")
        return max_length
    
    def minimum_parentheses_to_add(self, s: str) -> int:
        """
        Find minimum parentheses to add to make string valid
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(n), Space: O(1)
        """
        open_needed = 0  # Number of '(' needed
        close_needed = 0  # Number of ')' needed
        
        print(f"Finding minimum parentheses to add for: '{s}'")
        print()
        
        for i, char in enumerate(s):
            print(f"Step {i+1}: Processing '{char}'")
            
            if char == '(':
                close_needed += 1
                print(f"   Opening '(': Will need closing ')' later")
                print(f"   Close needed: {close_needed}")
            elif char == ')':
                if close_needed > 0:
                    close_needed -= 1
                    print(f"   Closing ')': Matched with previous '('")
                    print(f"   Close needed: {close_needed}")
                else:
                    open_needed += 1
                    print(f"   Closing ')': No matching '(', need opening")
                    print(f"   Open needed: {open_needed}")
            print()
        
        total_needed = open_needed + close_needed
        print(f"Total parentheses needed: {open_needed} + {close_needed} = {total_needed}")
        return total_needed
    
    # ==========================================
    # 4. STACK-BASED SORTING AND OPERATIONS
    # ==========================================
    
    def sort_stack(self, stack: List[int]) -> List[int]:
        """
        Sort stack using only stack operations
        
        Company: Microsoft, Amazon
        Difficulty: Medium
        Time: O(n²), Space: O(n)
        
        Algorithm: Use auxiliary stack to maintain sorted order
        """
        auxiliary_stack = []
        
        print(f"Sorting stack: {stack}")
        print("Using auxiliary stack to maintain sorted order")
        print()
        
        step = 1
        while stack:
            print(f"Step {step}: Main stack: {stack}, Aux stack: {auxiliary_stack}")
            
            # Pop element from main stack
            temp = stack.pop()
            print(f"   Pop {temp} from main stack")
            
            # Move elements from auxiliary to main until correct position
            moved_count = 0
            while auxiliary_stack and auxiliary_stack[-1] > temp:
                moved_element = auxiliary_stack.pop()
                stack.append(moved_element)
                moved_count += 1
                print(f"   Move {moved_element} from aux to main (>{temp})")
            
            # Push temp to auxiliary stack
            auxiliary_stack.append(temp)
            print(f"   Push {temp} to aux stack")
            print(f"   Result: Main: {stack}, Aux: {auxiliary_stack}")
            print()
            step += 1
        
        # Transfer sorted elements back to main stack
        print("Transferring sorted elements back to main stack:")
        while auxiliary_stack:
            element = auxiliary_stack.pop()
            stack.append(element)
            print(f"   Move {element} from aux to main")
        
        print(f"Final sorted stack: {stack}")
        return stack
    
    def stack_using_queues(self):
        """
        Implement stack using two queues
        
        Company: Amazon, Microsoft
        Difficulty: Easy
        Time: O(n) for pop, O(1) for push
        """
        
        class StackUsingQueues:
            def __init__(self):
                self.q1 = deque()
                self.q2 = deque()
            
            def push(self, x: int) -> None:
                print(f"Push {x}:")
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
                if not self.q1:
                    print("Stack is empty!")
                    return -1
                
                element = self.q1.popleft()
                print(f"Pop {element}: q1 becomes {list(self.q1)}")
                return element
            
            def top(self) -> int:
                if not self.q1:
                    print("Stack is empty!")
                    return -1
                
                element = self.q1[0]
                print(f"Top element: {element}")
                return element
            
            def empty(self) -> bool:
                is_empty = len(self.q1) == 0
                print(f"Is empty: {is_empty}")
                return is_empty
        
        print("=== STACK USING QUEUES DEMONSTRATION ===")
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
    
    # ==========================================
    # 5. ADVANCED STACK PROBLEMS
    # ==========================================
    
    def trapping_rain_water(self, height: List[int]) -> int:
        """
        Calculate trapped rainwater using stack
        
        Company: Amazon, Google, Microsoft
        Difficulty: Hard
        Time: O(n), Space: O(n)
        
        Algorithm: Use stack to find areas between bars
        """
        stack = []
        water_trapped = 0
        
        print(f"Calculating trapped rainwater for heights: {height}")
        print("Using stack to find water pockets between bars")
        print()
        
        for i in range(len(height)):
            print(f"Step {i+1}: Processing height[{i}] = {height[i]}")
            
            # Pop bars from stack while current bar is higher
            while stack and height[i] > height[stack[-1]]:
                top = stack.pop()
                print(f"   Current bar ({height[i]}) > stack top ({height[top]})")
                print(f"   Pop index {top} (height {height[top]})")
                
                if not stack:
                    print(f"   Stack empty, no left boundary")
                    break
                
                # Calculate trapped water
                distance = i - stack[-1] - 1
                bounded_height = min(height[i], height[stack[-1]]) - height[top]
                water = distance * bounded_height
                water_trapped += water
                
                print(f"   Left boundary: {stack[-1]} (height {height[stack[-1]]})")
                print(f"   Right boundary: {i} (height {height[i]})")
                print(f"   Bottom: {top} (height {height[top]})")
                print(f"   Distance: {distance}, Bounded height: {bounded_height}")
                print(f"   Water trapped: {water}")
                print(f"   Total water: {water_trapped}")
            
            # Push current index to stack
            stack.append(i)
            print(f"   Push index {i} to stack: {stack}")
            print()
        
        print(f"Total trapped rainwater: {water_trapped}")
        return water_trapped
    
    def asteroid_collision(self, asteroids: List[int]) -> List[int]:
        """
        Simulate asteroid collisions using stack
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(n), Space: O(n)
        
        Rules:
        - Positive = moving right, Negative = moving left
        - Collisions occur when right-moving meets left-moving
        - Larger asteroid survives, equal sizes both explode
        """
        stack = []
        
        print(f"Simulating asteroid collisions: {asteroids}")
        print("Positive = right, Negative = left")
        print()
        
        for i, asteroid in enumerate(asteroids):
            print(f"Step {i+1}: Processing asteroid {asteroid}")
            
            # Check for collisions
            while (stack and asteroid < 0 and stack[-1] > 0):
                # Collision occurs
                top = stack[-1]
                print(f"   Collision: {top} (right) vs {asteroid} (left)")
                
                if abs(asteroid) > top:
                    # Current asteroid destroys the one in stack
                    stack.pop()
                    print(f"   {asteroid} destroys {top}")
                    print(f"   Stack after destruction: {stack}")
                elif abs(asteroid) == top:
                    # Both asteroids explode
                    stack.pop()
                    asteroid = 0  # Mark as destroyed
                    print(f"   Both asteroids explode")
                    print(f"   Stack after explosion: {stack}")
                    break
                else:
                    # Stack asteroid survives
                    asteroid = 0  # Mark current as destroyed
                    print(f"   {top} survives, {asteroid} is destroyed")
                    break
            
            # Add asteroid to stack if it survived
            if asteroid != 0:
                stack.append(asteroid)
                print(f"   Add surviving asteroid {asteroid} to stack")
            
            print(f"   Current stack: {stack}")
            print()
        
        print(f"Final surviving asteroids: {stack}")
        return stack


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_stack_algorithms():
    """Demonstrate all stack algorithms and applications"""
    print("=== STACK ALGORITHMS DEMONSTRATION ===\n")
    
    algorithms = StackAlgorithms()
    
    # 1. Monotonic stack problems
    print("=== MONOTONIC STACK PROBLEMS ===")
    
    print("1. Next Greater Elements:")
    algorithms.next_greater_elements([2, 1, 2, 4, 3, 1])
    print("\n" + "-"*40 + "\n")
    
    print("2. Previous Smaller Elements:")
    algorithms.previous_smaller_elements([4, 5, 2, 10, 8])
    print("\n" + "-"*40 + "\n")
    
    print("3. Daily Temperatures:")
    algorithms.daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])
    print("\n" + "="*60 + "\n")
    
    # 2. Histogram problems
    print("=== HISTOGRAM PROBLEMS ===")
    
    print("1. Largest Rectangle in Histogram:")
    algorithms.largest_rectangle_in_histogram([2, 1, 5, 6, 2, 3])
    print("\n" + "-"*40 + "\n")
    
    print("2. Maximal Rectangle in Binary Matrix:")
    matrix = [
        ["1", "0", "1", "0", "0"],
        ["1", "0", "1", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "0", "0", "1", "0"]
    ]
    algorithms.maximal_rectangle_binary_matrix(matrix)
    print("\n" + "="*60 + "\n")
    
    # 3. Parentheses problems
    print("=== PARENTHESES PROBLEMS ===")
    
    print("1. Longest Valid Parentheses:")
    algorithms.longest_valid_parentheses("(()")
    print("\n" + "-"*40 + "\n")
    
    print("2. Minimum Parentheses to Add:")
    algorithms.minimum_parentheses_to_add("())")
    print("\n" + "="*60 + "\n")
    
    # 4. Stack operations
    print("=== STACK OPERATIONS ===")
    
    print("1. Sort Stack:")
    test_stack = [34, 3, 31, 98, 92, 23]
    algorithms.sort_stack(test_stack)
    print("\n" + "-"*40 + "\n")
    
    print("2. Stack Using Queues:")
    algorithms.stack_using_queues()
    print("\n" + "="*60 + "\n")
    
    # 5. Advanced problems
    print("=== ADVANCED STACK PROBLEMS ===")
    
    print("1. Trapping Rain Water:")
    algorithms.trapping_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
    print("\n" + "-"*40 + "\n")
    
    print("2. Asteroid Collision:")
    algorithms.asteroid_collision([5, 10, -5])
    print()


if __name__ == "__main__":
    demonstrate_stack_algorithms()
    
    print("=== STACK ALGORITHMS MASTERY GUIDE ===")
    
    print("\n🎯 MONOTONIC STACK PATTERNS:")
    print("• Monotonic Increasing: For 'previous smaller' problems")
    print("• Monotonic Decreasing: For 'next greater' problems")
    print("• Store indices instead of values for position tracking")
    print("• O(n) time complexity despite nested loops")
    
    print("\n📊 HISTOGRAM TECHNIQUES:")
    print("• Use stack to maintain increasing heights")
    print("• Calculate area when height decreases")
    print("• Width = current_index - previous_index - 1")
    print("• Essential for rectangle and matrix problems")
    
    print("\n🔧 OPTIMIZATION STRATEGIES:")
    print("• Amortized analysis: Each element pushed/popped once")
    print("• Space-time tradeoffs with auxiliary stacks")
    print("• Combine with other data structures when needed")
    print("• Consider iterative vs recursive approaches")
    
    print("\n⚡ PROBLEM IDENTIFICATION:")
    print("• 'Next/Previous Greater/Smaller' → Monotonic Stack")
    print("• 'Largest Rectangle/Area' → Histogram Stack")
    print("• 'Valid Parentheses/Brackets' → Matching Stack")
    print("• 'Trapped Water/Heights' → Stack with Area Calculation")
    
    print("\n🏆 INTERVIEW SUCCESS TIPS:")
    print("• Master the monotonic stack template")
    print("• Practice histogram area calculations")
    print("• Understand when to store values vs indices")
    print("• Draw examples to visualize stack operations")
    print("• Consider edge cases: empty arrays, single elements"))

