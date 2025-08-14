"""
Stack Interview Problems - Most Asked Questions
===============================================

Topics: Common stack interview questions with solutions and explanations
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber, LinkedIn
Difficulty: Easy to Hard
Time Complexity: Usually O(n) with stack optimization
Space Complexity: O(n) for stack storage
"""

from typing import List, Optional, Dict, Tuple, Any
from collections import deque

class StackInterviewProblems:
    
    def __init__(self):
        """Initialize with problem tracking"""
        self.problem_count = 0
        self.solution_steps = []
    
    # ==========================================
    # 1. CLASSIC STACK INTERVIEW PROBLEMS
    # ==========================================
    
    def valid_parentheses(self, s: str) -> bool:
        """
        Valid Parentheses - Most Asked Stack Problem
        
        Check if string has valid parentheses: ()[]{}
        
        Company: Almost every tech company
        Difficulty: Easy
        Time: O(n), Space: O(n)
        
        Example: "()" → True, "([{}])" → True, "([)]" → False
        """
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        print(f"Checking valid parentheses: '{s}'")
        print(f"Mapping: {mapping}")
        print()
        
        for i, char in enumerate(s):
            print(f"Step {i+1}: Processing '{char}'")
            
            if char in mapping:
                # Closing bracket
                if not stack:
                    print(f"   ✗ Closing '{char}' without opening bracket")
                    return False
                
                top_element = stack.pop()
                if mapping[char] != top_element:
                    print(f"   ✗ Mismatch: '{top_element}' vs expected '{mapping[char]}'")
                    return False
                
                print(f"   ✓ Matched '{top_element}' with '{char}'")
                print(f"   Stack after pop: {stack}")
            else:
                # Opening bracket
                stack.append(char)
                print(f"   Push opening '{char}' to stack")
                print(f"   Stack: {stack}")
        
        is_valid = len(stack) == 0
        print(f"\nResult: {'Valid' if is_valid else 'Invalid'}")
        if not is_valid:
            print(f"Unmatched opening brackets: {stack}")
        
        return is_valid
    
    def min_stack_design(self):
        """
        Design Min Stack - Get minimum in O(1)
        
        Company: Amazon, Microsoft, Google
        Difficulty: Easy
        Time: O(1) for all operations, Space: O(n)
        
        Support: push, pop, top, getMin in O(1)
        """
        
        class MinStack:
            def __init__(self):
                self.stack = []
                self.min_stack = []  # Stack to track minimums
                self.operation_count = 0
            
            def push(self, val: int) -> None:
                self.operation_count += 1
                print(f"Operation {self.operation_count}: Push {val}")
                
                self.stack.append(val)
                
                # Update min_stack
                if not self.min_stack or val <= self.min_stack[-1]:
                    self.min_stack.append(val)
                    print(f"   New minimum: {val}")
                
                print(f"   Main stack: {self.stack}")
                print(f"   Min stack: {self.min_stack}")
                print(f"   Current min: {self.getMin()}")
            
            def pop(self) -> None:
                self.operation_count += 1
                print(f"Operation {self.operation_count}: Pop")
                
                if not self.stack:
                    print("   ✗ Stack is empty")
                    return
                
                val = self.stack.pop()
                print(f"   Popped: {val}")
                
                # Update min_stack if popped element was minimum
                if self.min_stack and val == self.min_stack[-1]:
                    self.min_stack.pop()
                    print(f"   Removed from min stack")
                
                print(f"   Main stack: {self.stack}")
                print(f"   Min stack: {self.min_stack}")
                if self.stack:
                    print(f"   Current min: {self.getMin()}")
            
            def top(self) -> int:
                if not self.stack:
                    print("   ✗ Stack is empty")
                    return -1
                return self.stack[-1]
            
            def getMin(self) -> int:
                if not self.min_stack:
                    print("   ✗ No minimum (empty stack)")
                    return -1
                return self.min_stack[-1]
        
        print("=== MIN STACK DESIGN ===")
        min_stack = MinStack()
        
        # Test operations
        operations = [
            ('push', -2), ('push', 0), ('push', -3),
            ('getMin', None), ('pop', None), ('top', None), ('getMin', None)
        ]
        
        for op, val in operations:
            print()
            if op == 'push':
                min_stack.push(val)
            elif op == 'pop':
                min_stack.pop()
            elif op == 'top':
                result = min_stack.top()
                print(f"Top element: {result}")
            elif op == 'getMin':
                result = min_stack.getMin()
                print(f"Minimum element: {result}")
        
        return min_stack
    
    def evaluate_reverse_polish_notation(self, tokens: List[str]) -> int:
        """
        Evaluate Reverse Polish Notation (RPN)
        
        Company: Amazon, Microsoft, Facebook
        Difficulty: Medium
        Time: O(n), Space: O(n)
        
        Example: ["2","1","+","3","*"] → ((2+1)*3) = 9
        """
        stack = []
        operators = {'+', '-', '*', '/'}
        
        print(f"Evaluating RPN: {tokens}")
        print()
        
        for i, token in enumerate(tokens):
            print(f"Step {i+1}: Processing '{token}'")
            
            if token in operators:
                # Pop two operands
                if len(stack) < 2:
                    raise ValueError("Invalid RPN expression")
                
                b = stack.pop()
                a = stack.pop()
                
                print(f"   Operator '{token}': Pop {b} and {a}")
                
                # Perform operation
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    result = int(a / b)  # Truncate toward zero
                
                stack.append(result)
                print(f"   Calculate: {a} {token} {b} = {result}")
                print(f"   Stack: {stack}")
            else:
                # Operand
                num = int(token)
                stack.append(num)
                print(f"   Operand: Push {num}")
                print(f"   Stack: {stack}")
        
        if len(stack) != 1:
            raise ValueError("Invalid RPN expression")
        
        result = stack[0]
        print(f"\nFinal result: {result}")
        return result
    
    def decode_string(self, s: str) -> str:
        """
        Decode String using Stack
        
        Company: Google, Facebook, Amazon
        Difficulty: Medium
        Time: O(n), Space: O(n)
        
        Example: "3[a2[c]]" → "accaccacc"
        """
        stack = []
        current_num = 0
        current_string = ""
        
        print(f"Decoding string: '{s}'")
        print()
        
        for i, char in enumerate(s):
            print(f"Step {i+1}: Processing '{char}'")
            
            if char.isdigit():
                current_num = current_num * 10 + int(char)
                print(f"   Digit: Building number = {current_num}")
            
            elif char == '[':
                # Push current state to stack
                stack.append((current_string, current_num))
                print(f"   '[': Push state ({current_string}, {current_num}) to stack")
                
                # Reset for new context
                current_string = ""
                current_num = 0
                print(f"   Reset: string='', num=0")
                print(f"   Stack: {stack}")
            
            elif char == ']':
                # Pop from stack and build string
                prev_string, num = stack.pop()
                print(f"   ']': Pop state ({prev_string}, {num}) from stack")
                
                # Build repeated string
                repeated = current_string * num
                current_string = prev_string + repeated
                
                print(f"   Repeat '{current_string[len(prev_string):]}' {num} times")
                print(f"   Result: '{prev_string}' + '{repeated}' = '{current_string}'")
                print(f"   Stack: {stack}")
            
            else:
                # Regular character
                current_string += char
                print(f"   Character: Add '{char}' to string = '{current_string}'")
        
        print(f"\nFinal decoded string: '{current_string}'")
        return current_string
    
    # ==========================================
    # 2. MEDIUM DIFFICULTY PROBLEMS
    # ==========================================
    
    def basic_calculator(self, s: str) -> int:
        """
        Basic Calculator - Handle +, -, (, )
        
        Company: Google, Amazon, Microsoft
        Difficulty: Hard
        Time: O(n), Space: O(n)
        
        Example: "(1+(4+5+2)-3)+(6+8)" → 23
        """
        stack = []
        current_num = 0
        result = 0
        sign = 1  # 1 for positive, -1 for negative
        
        print(f"Calculating: '{s}'")
        print("Using stack to handle parentheses")
        print()
        
        for i, char in enumerate(s):
            if char == ' ':
                continue
            
            print(f"Step {i+1}: Processing '{char}'")
            
            if char.isdigit():
                current_num = current_num * 10 + int(char)
                print(f"   Digit: Building number = {current_num}")
            
            elif char in ['+', '-']:
                # Process previous number
                result += sign * current_num
                print(f"   Operator '{char}': Add {sign * current_num} to result = {result}")
                
                # Set sign for next number
                sign = 1 if char == '+' else -1
                current_num = 0
                print(f"   Next sign: {sign}")
            
            elif char == '(':
                # Push current result and sign to stack
                stack.append(result)
                stack.append(sign)
                print(f"   '(': Push result={result}, sign={sign} to stack")
                print(f"   Stack: {stack}")
                
                # Reset for expression inside parentheses
                result = 0
                sign = 1
                print(f"   Reset: result=0, sign=1")
            
            elif char == ')':
                # Process current number
                result += sign * current_num
                print(f"   ')': Add {sign * current_num} to result = {result}")
                
                # Pop sign and previous result
                prev_sign = stack.pop()
                prev_result = stack.pop()
                print(f"   Pop: prev_sign={prev_sign}, prev_result={prev_result}")
                
                # Combine with previous result
                result = prev_result + prev_sign * result
                print(f"   Combine: {prev_result} + {prev_sign} * {result // prev_sign} = {result}")
                print(f"   Stack: {stack}")
                
                current_num = 0
        
        # Process final number
        result += sign * current_num
        print(f"\nAdd final number: {sign * current_num}")
        print(f"Final result: {result}")
        
        return result
    
    def remove_k_digits(self, num: str, k: int) -> str:
        """
        Remove K Digits to Make Smallest Number
        
        Company: Amazon, Google, Microsoft
        Difficulty: Medium
        Time: O(n), Space: O(n)
        
        Example: num="1432219", k=3 → "1219"
        """
        stack = []
        to_remove = k
        
        print(f"Remove {k} digits from '{num}' to make smallest number")
        print("Using monotonic increasing stack strategy")
        print()
        
        for i, digit in enumerate(num):
            print(f"Step {i+1}: Processing digit '{digit}'")
            
            # Remove larger digits from stack to make number smaller
            while stack and to_remove > 0 and stack[-1] > digit:
                removed = stack.pop()
                to_remove -= 1
                print(f"   Remove '{removed}' (larger than {digit})")
                print(f"   Remaining removals: {to_remove}")
                print(f"   Stack: {stack}")
            
            stack.append(digit)
            print(f"   Add '{digit}' to stack: {stack}")
        
        # If we still need to remove digits, remove from end
        while to_remove > 0:
            removed = stack.pop()
            to_remove -= 1
            print(f"   Remove '{removed}' from end")
        
        # Build result, removing leading zeros
        result = ''.join(stack).lstrip('0')
        if not result:
            result = '0'
        
        print(f"\nSmallest number after removing {k} digits: '{result}'")
        return result
    
    def exclusive_time_functions(self, n: int, logs: List[str]) -> List[int]:
        """
        Exclusive Time of Functions
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(m), Space: O(n) where m is number of logs
        
        Calculate exclusive execution time for each function
        """
        stack = []
        result = [0] * n
        
        print(f"Calculating exclusive time for {n} functions")
        print(f"Logs: {logs}")
        print()
        
        for i, log in enumerate(logs):
            function_id, action, timestamp = log.split(':')
            function_id = int(function_id)
            timestamp = int(timestamp)
            
            print(f"Step {i+1}: Processing {log}")
            
            if action == 'start':
                # Function starts
                stack.append((function_id, timestamp))
                print(f"   Function {function_id} starts at time {timestamp}")
                print(f"   Stack: {stack}")
            
            else:  # action == 'end'
                # Function ends
                start_function, start_time = stack.pop()
                execution_time = timestamp - start_time + 1
                result[function_id] += execution_time
                
                print(f"   Function {function_id} ends at time {timestamp}")
                print(f"   Execution time: {timestamp} - {start_time} + 1 = {execution_time}")
                print(f"   Total time for function {function_id}: {result[function_id]}")
                
                # Subtract this time from parent function
                if stack:
                    # Parent function was running during this time
                    parent_function = stack[-1][0]
                    result[parent_function] -= execution_time
                    print(f"   Subtract {execution_time} from parent function {parent_function}")
                
                print(f"   Stack: {stack}")
                print(f"   Current results: {result}")
        
        print(f"\nFinal exclusive times: {result}")
        return result
    
    # ==========================================
    # 3. HARD DIFFICULTY PROBLEMS
    # ==========================================
    
    def largest_rectangle_histogram(self, heights: List[int]) -> int:
        """
        Largest Rectangle in Histogram - Classic Hard Problem
        
        Company: Google, Amazon, Microsoft
        Difficulty: Hard
        Time: O(n), Space: O(n)
        
        Find area of largest rectangle that can be formed in histogram
        """
        stack = []
        max_area = 0
        index = 0
        
        print(f"Finding largest rectangle in histogram: {heights}")
        print("Using stack to track increasing heights")
        print()
        
        while index < len(heights):
            print(f"Step {index + 1}: Processing height[{index}] = {heights[index]}")
            
            # If current bar is higher, push to stack
            if not stack or heights[index] >= heights[stack[-1]]:
                stack.append(index)
                print(f"   Height is non-decreasing, push index {index}")
                print(f"   Stack: {stack}")
                index += 1
            else:
                # Current bar is lower, calculate area with popped bar
                top_index = stack.pop()
                height = heights[top_index]
                
                # Calculate width
                width = index if not stack else index - stack[-1] - 1
                area = height * width
                max_area = max(max_area, area)
                
                print(f"   Height decreases, pop index {top_index}")
                print(f"   Calculate area: height={height}, width={width}")
                print(f"   Area = {height} × {width} = {area}")
                print(f"   Max area so far: {max_area}")
                print(f"   Stack: {stack}")
        
        # Process remaining bars in stack
        print(f"\nProcessing remaining bars:")
        while stack:
            top_index = stack.pop()
            height = heights[top_index]
            width = index if not stack else index - stack[-1] - 1
            area = height * width
            max_area = max(max_area, area)
            
            print(f"   Pop index {top_index}: height={height}, width={width}, area={area}")
            print(f"   Max area: {max_area}")
        
        print(f"\nLargest rectangle area: {max_area}")
        return max_area
    
    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        """
        Sliding Window Maximum using Deque (Stack-like operations)
        
        Company: Amazon, Google, Microsoft
        Difficulty: Hard
        Time: O(n), Space: O(k)
        
        Find maximum in each sliding window of size k
        """
        from collections import deque
        
        dq = deque()  # Store indices
        result = []
        
        print(f"Finding sliding window maximum: nums={nums}, k={k}")
        print("Using deque with stack-like operations for optimization")
        print()
        
        for i in range(len(nums)):
            print(f"Step {i+1}: Processing nums[{i}] = {nums[i]}")
            
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                removed_idx = dq.popleft()
                print(f"   Remove index {removed_idx} (outside window)")
            
            # Remove indices with smaller values (they can't be maximum)
            while dq and nums[dq[-1]] < nums[i]:
                removed_idx = dq.pop()
                print(f"   Remove index {removed_idx} (nums[{removed_idx}]={nums[removed_idx]} < {nums[i]})")
            
            # Add current index
            dq.append(i)
            print(f"   Add index {i}")
            print(f"   Deque indices: {list(dq)}")
            print(f"   Deque values: {[nums[idx] for idx in dq]}")
            
            # If window is complete, record maximum
            if i >= k - 1:
                maximum = nums[dq[0]]
                result.append(maximum)
                print(f"   Window [{i-k+1}:{i+1}]: Maximum = {maximum}")
            
            print()
        
        print(f"Sliding window maximums: {result}")
        return result
    
    # ==========================================
    # 4. INTERVIEW STRATEGY AND TIPS
    # ==========================================
    
    def interview_approach_guide(self) -> None:
        """
        Comprehensive guide for tackling stack interview problems
        """
        print("=== STACK INTERVIEW APPROACH GUIDE ===")
        print()
        
        print("🎯 STEP 1: PROBLEM RECOGNITION")
        print("Look for these keywords/patterns:")
        print("• 'Parentheses', 'brackets', 'balanced'")
        print("• 'Next/Previous greater/smaller'")
        print("• 'Nearest', 'closest', 'immediate'")
        print("• 'Valid', 'matching', 'corresponding'")
        print("• 'Nested', 'inside-out processing'")
        print("• 'Undo', 'backtrack', 'reverse'")
        print()
        
        print("🎯 STEP 2: CHOOSE STACK APPROACH")
        print("Decision framework:")
        print("• LIFO needed? → Stack")
        print("• Matching pairs? → Stack with mapping")
        print("• Monotonic property? → Monotonic stack")
        print("• Expression parsing? → Two stacks or conversion")
        print("• Nested structures? → Stack for context management")
        print()
        
        print("🎯 STEP 3: IMPLEMENTATION TEMPLATE")
        print("Common patterns:")
        print()
        print("A) Basic Stack Template:")
        print("   stack = []")
        print("   for element in input:")
        print("       # Process based on conditions")
        print("       # Push/pop based on logic")
        print()
        print("B) Monotonic Stack Template:")
        print("   stack = []")
        print("   for i, element in enumerate(input):")
        print("       while stack and condition(stack[-1], element):")
        print("           # Process popped element")
        print("           stack.pop()")
        print("       stack.append(i or element)")
        print()
        print("C) Matching Template:")
        print("   stack = []")
        print("   mapping = {closing: opening}")
        print("   for char in string:")
        print("       if char in opening:")
        print("           stack.append(char)")
        print("       elif char in closing:")
        print("           # Check matching")
        print()
        
        print("🎯 STEP 4: OPTIMIZATION CONSIDERATIONS")
        print("• Space optimization: Can you use indices instead of values?")
        print("• Time optimization: Is the stack operation really O(1) amortized?")
        print("• Alternative approaches: Can you solve without stack?")
        print("• Edge cases: Empty input, single element, all same elements")
        print()
        
        print("🎯 STEP 5: TESTING STRATEGY")
        print("Test cases to consider:")
        print("• Empty input")
        print("• Single element")
        print("• Already sorted/optimal")
        print("• Worst case scenario")
        print("• Typical case")
        print("• Edge boundaries")
    
    def common_mistakes(self) -> None:
        """
        Common mistakes in stack interview problems
        """
        print("=== COMMON STACK INTERVIEW MISTAKES ===")
        print()
        
        print("❌ MISTAKE 1: Not checking empty stack before pop/peek")
        print("Problem: Runtime error when stack is empty")
        print("Solution: Always check 'if stack:' before accessing")
        print()
        
        print("❌ MISTAKE 2: Using wrong data structure")
        print("Problem: Using stack when queue is needed or vice versa")
        print("Solution: Understand LIFO vs FIFO requirements")
        print()
        
        print("❌ MISTAKE 3: Storing wrong information in stack")
        print("Problem: Storing values when indices are needed")
        print("Solution: Think about what information you need later")
        print()
        
        print("❌ MISTAKE 4: Incorrect monotonic stack direction")
        print("Problem: Using increasing when decreasing is needed")
        print("Solution: Draw examples to understand the pattern")
        print()
        
        print("❌ MISTAKE 5: Not handling edge cases")
        print("Problem: Fails on empty input or single elements")
        print("Solution: Test with minimal inputs first")
        print()
        
        print("❌ MISTAKE 6: Overcomplicating the solution")
        print("Problem: Adding unnecessary logic or data structures")
        print("Solution: Start with simplest approach that works")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_stack_interview_problems():
    """Demonstrate key stack interview problems"""
    print("=== STACK INTERVIEW PROBLEMS DEMONSTRATION ===\n")
    
    problems = StackInterviewProblems()
    
    # 1. Classic problems
    print("=== CLASSIC STACK PROBLEMS ===")
    
    print("1. Valid Parentheses:")
    test_cases = ["()", "()[]{}","(]", "([)]", "{[]}"]
    for case in test_cases:
        result = problems.valid_parentheses(case)
        print()
    
    print("-" * 60 + "\n")
    
    print("2. Min Stack Design:")
    problems.min_stack_design()
    print("\n" + "-" * 60 + "\n")
    
    print("3. Evaluate RPN:")
    problems.evaluate_reverse_polish_notation(["2","1","+","3","*"])
    print("\n" + "-" * 60 + "\n")
    
    print("4. Decode String:")
    problems.decode_string("3[a2[c]]")
    print("\n" + "=" * 60 + "\n")
    
    # 2. Medium problems
    print("=== MEDIUM DIFFICULTY PROBLEMS ===")
    
    print("1. Basic Calculator:")
    problems.basic_calculator("(1+(4+5+2)-3)+(6+8)")
    print("\n" + "-" * 60 + "\n")
    
    print("2. Remove K Digits:")
    problems.remove_k_digits("1432219", 3)
    print("\n" + "-" * 60 + "\n")
    
    print("3. Exclusive Time of Functions:")
    logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
    problems.exclusive_time_functions(2, logs)
    print("\n" + "=" * 60 + "\n")
    
    # 3. Hard problems
    print("=== HARD DIFFICULTY PROBLEMS ===")
    
    print("1. Largest Rectangle in Histogram:")
    problems.largest_rectangle_histogram([2,1,5,6,2,3])
    print("\n" + "-" * 60 + "\n")
    
    print("2. Sliding Window Maximum:")
    problems.sliding_window_maximum([1,3,-1,-3,5,3,6,7], 3)
    print("\n" + "=" * 60 + "\n")
    
    # 4. Interview guidance
    problems.interview_approach_guide()
    print("\n" + "=" * 60 + "\n")
    
    problems.common_mistakes()


if __name__ == "__main__":
    demonstrate_stack_interview_problems()
    
    print("\n=== STACK INTERVIEW SUCCESS STRATEGY ===")
    
    print("\n🎯 PREPARATION ROADMAP:")
    print("Week 1: Master basic stack operations and simple problems")
    print("Week 2: Practice monotonic stack and histogram problems")
    print("Week 3: Tackle expression evaluation and parsing")
    print("Week 4: Solve advanced problems and optimize solutions")
    
    print("\n📚 MUST-PRACTICE PROBLEMS:")
    print("• Valid Parentheses (Easy)")
    print("• Min Stack Design (Easy)")
    print("• Next Greater Element (Medium)")
    print("• Daily Temperatures (Medium)")
    print("• Largest Rectangle in Histogram (Hard)")
    print("• Basic Calculator (Hard)")
    print("• Decode String (Medium)")
    print("• Remove K Digits (Medium)")
    
    print("\n⚡ QUICK PROBLEM IDENTIFICATION:")
    print("• Balanced/Matching → Stack with hashmap")
    print("• Next/Previous Greater → Monotonic stack")
    print("• Expression Evaluation → Stack or conversion")
    print("• Nested Processing → Stack for context")
    print("• LIFO Behavior → Direct stack application")
    
    print("\n🏆 INTERVIEW DAY TIPS:")
    print("• Clarify input constraints and edge cases")
    print("• Draw examples to visualize stack operations")
    print("• Start with brute force, then optimize")
    print("• Explain your approach before coding")
    print("• Test with edge cases after implementation")
    print("• Discuss time/space complexity")
    
    print("\n📊 COMPLEXITY GOALS:")
    print("• Time: O(n) for most stack problems")
    print("• Space: O(n) for stack storage")
    print("• Amortized analysis: Each element pushed/popped once")
    print("• Monotonic stack: Despite nested loops, still O(n)")
    
    print("\n🎓 ADVANCED PREPARATION:")
    print("• Study different stack implementations")
    print("• Practice with memory constraints")
    print("• Learn stack-based algorithms (DFS, expression parsing)")
    print("• Understand when NOT to use stack")
    print("• Practice explaining solutions clearly")

