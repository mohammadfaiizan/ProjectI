"""
32. Longest Valid Parentheses - Multiple Approaches
Difficulty: Hard

Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.
"""

from typing import List

class LongestValidParentheses:
    """Multiple approaches to find longest valid parentheses substring"""
    
    def longestValidParentheses_stack_approach(self, s: str) -> int:
        """
        Approach 1: Stack-based Approach
        
        Use stack to track indices and find valid parentheses.
        
        Time: O(n), Space: O(n)
        """
        stack = [-1]  # Initialize with -1 as base
        max_length = 0
        
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            else:  # char == ')'
                stack.pop()
                
                if not stack:
                    # No matching '(', push current index as new base
                    stack.append(i)
                else:
                    # Calculate length of current valid substring
                    current_length = i - stack[-1]
                    max_length = max(max_length, current_length)
        
        return max_length
    
    def longestValidParentheses_dp_approach(self, s: str) -> int:
        """
        Approach 2: Dynamic Programming Approach
        
        Use DP to track length of valid parentheses ending at each position.
        
        Time: O(n), Space: O(n)
        """
        if not s:
            return 0
        
        n = len(s)
        dp = [0] * n  # dp[i] = length of valid parentheses ending at index i
        max_length = 0
        
        for i in range(1, n):
            if s[i] == ')':
                if s[i-1] == '(':
                    # Case: ...()
                    dp[i] = (dp[i-2] if i >= 2 else 0) + 2
                elif dp[i-1] > 0:
                    # Case: ...))
                    # Check if there's a matching '(' for current ')'
                    match_index = i - dp[i-1] - 1
                    if match_index >= 0 and s[match_index] == '(':
                        dp[i] = dp[i-1] + 2 + (dp[match_index-1] if match_index > 0 else 0)
                
                max_length = max(max_length, dp[i])
        
        return max_length
    
    def longestValidParentheses_two_pass_approach(self, s: str) -> int:
        """
        Approach 3: Two-pass Counting Approach
        
        Count parentheses in both directions.
        
        Time: O(n), Space: O(1)
        """
        def count_parentheses(s: str, left_char: str, right_char: str) -> int:
            left_count = right_count = max_length = 0
            
            for char in s:
                if char == left_char:
                    left_count += 1
                elif char == right_char:
                    right_count += 1
                
                if left_count == right_count:
                    max_length = max(max_length, 2 * right_count)
                elif right_count > left_count:
                    left_count = right_count = 0
            
            return max_length
        
        # Left to right pass
        left_to_right = count_parentheses(s, '(', ')')
        
        # Right to left pass
        right_to_left = count_parentheses(s[::-1], ')', '(')
        
        return max(left_to_right, right_to_left)
    
    def longestValidParentheses_brute_force(self, s: str) -> int:
        """
        Approach 4: Brute Force Approach
        
        Check all possible substrings.
        
        Time: O(n³), Space: O(n)
        """
        def is_valid(substring: str) -> bool:
            """Check if substring has valid parentheses"""
            stack = []
            for char in substring:
                if char == '(':
                    stack.append(char)
                elif char == ')':
                    if not stack:
                        return False
                    stack.pop()
            return len(stack) == 0
        
        max_length = 0
        n = len(s)
        
        # Check all even-length substrings
        for i in range(n):
            for j in range(i + 2, n + 1, 2):  # Only even lengths
                if is_valid(s[i:j]):
                    max_length = max(max_length, j - i)
        
        return max_length
    
    def longestValidParentheses_optimized_stack(self, s: str) -> int:
        """
        Approach 5: Optimized Stack with Indices
        
        Track both characters and their indices efficiently.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        valid = [False] * len(s)
        
        # Mark all valid parentheses
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                # Mark both positions as valid
                left_index = stack.pop()
                valid[left_index] = True
                valid[i] = True
        
        # Find longest consecutive valid sequence
        max_length = current_length = 0
        
        for is_valid_char in valid:
            if is_valid_char:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        
        return max_length
    
    def longestValidParentheses_recursive_approach(self, s: str) -> int:
        """
        Approach 6: Recursive Approach with Memoization
        
        Use recursion to find valid parentheses.
        
        Time: O(n²), Space: O(n²)
        """
        memo = {}
        
        def find_longest(start: int, end: int) -> int:
            """Find longest valid parentheses in range [start, end)"""
            if start >= end or (start, end) in memo:
                return memo.get((start, end), 0)
            
            max_length = 0
            
            # Try all possible splits
            for i in range(start, end):
                # Check if we can form a valid pair starting at 'start'
                if s[start] == '(' and s[i] == ')':
                    # Check if substring between start+1 and i is valid
                    inner_length = find_longest(start + 1, i)
                    if inner_length == i - start - 1:  # All characters in between are valid
                        # Add this pair and check remaining
                        remaining_length = find_longest(i + 1, end)
                        total_length = 2 + inner_length + remaining_length
                        max_length = max(max_length, total_length)
                
                # Try skipping current character
                skip_length = find_longest(start + 1, end)
                max_length = max(max_length, skip_length)
            
            memo[(start, end)] = max_length
            return max_length
        
        return find_longest(0, len(s))
    
    def longestValidParentheses_segment_tree_approach(self, s: str) -> int:
        """
        Approach 7: Segment Tree Approach (Educational)
        
        Use segment tree for range queries (overkill for this problem).
        
        Time: O(n log n), Space: O(n)
        """
        # This is an educational approach - segment tree is overkill here
        # Fall back to the efficient DP approach
        return self.longestValidParentheses_dp_approach(s)

def test_longest_valid_parentheses():
    """Test longest valid parentheses algorithms"""
    solver = LongestValidParentheses()
    
    test_cases = [
        ("(()", 2, "Simple case"),
        (")()())", 4, "Multiple valid pairs"),
        ("", 0, "Empty string"),
        ("((()))", 6, "All valid nested"),
        ("()(()", 2, "Mixed valid/invalid"),
        ("()(())", 6, "All valid mixed"),
        ("((())())", 8, "Complex nested"),
        ("())", 2, "Extra closing"),
        ("(((", 0, "Only opening"),
        (")))", 0, "Only closing"),
        ("()()", 4, "Sequential pairs"),
        ("(()()", 4, "Partial nesting"),
        ("()()()()())", 10, "Long sequence"),
        ("(()())", 6, "Nested with pairs"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.longestValidParentheses_stack_approach),
        ("DP Approach", solver.longestValidParentheses_dp_approach),
        ("Two Pass", solver.longestValidParentheses_two_pass_approach),
        ("Optimized Stack", solver.longestValidParentheses_optimized_stack),
    ]
    
    print("=== Testing Longest Valid Parentheses ===")
    
    for s, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input:    '{s}'")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")

def demonstrate_dp_approach():
    """Demonstrate DP approach step by step"""
    print("\n=== DP Approach Step-by-Step Demo ===")
    
    s = "((()))"
    print(f"Finding longest valid parentheses in: '{s}'")
    
    n = len(s)
    dp = [0] * n
    max_length = 0
    
    print(f"\nInitial DP array: {dp}")
    
    for i in range(1, n):
        char = s[i]
        print(f"\nStep {i}: Processing s[{i}] = '{char}'")
        
        if char == ')':
            if s[i-1] == '(':
                # Case: ...()
                dp[i] = (dp[i-2] if i >= 2 else 0) + 2
                print(f"  -> Found '()' pattern")
                print(f"  -> dp[{i}] = dp[{i-2}] + 2 = {dp[i-2] if i >= 2 else 0} + 2 = {dp[i]}")
            elif dp[i-1] > 0:
                # Case: ...))
                match_index = i - dp[i-1] - 1
                print(f"  -> Checking for matching '(' at index {match_index}")
                
                if match_index >= 0 and s[match_index] == '(':
                    prev_length = dp[match_index-1] if match_index > 0 else 0
                    dp[i] = dp[i-1] + 2 + prev_length
                    print(f"  -> Found matching '(' at index {match_index}")
                    print(f"  -> dp[{i}] = dp[{i-1}] + 2 + dp[{match_index-1}] = {dp[i-1]} + 2 + {prev_length} = {dp[i]}")
                else:
                    print(f"  -> No matching '(' found")
            
            max_length = max(max_length, dp[i])
        
        print(f"  -> DP array: {dp}")
        print(f"  -> Max length so far: {max_length}")
    
    print(f"\nFinal result: {max_length}")

def visualize_stack_approach():
    """Visualize stack approach"""
    print("\n=== Stack Approach Visualization ===")
    
    s = ")()())"
    print(f"Processing: '{s}'")
    
    stack = [-1]
    max_length = 0
    
    print(f"Initial stack: {stack}")
    
    for i, char in enumerate(s):
        print(f"\nStep {i+1}: Processing s[{i}] = '{char}'")
        
        if char == '(':
            stack.append(i)
            print(f"  -> Push index {i} to stack")
        else:  # char == ')'
            stack.pop()
            print(f"  -> Pop from stack")
            
            if not stack:
                stack.append(i)
                print(f"  -> Stack empty, push {i} as new base")
            else:
                current_length = i - stack[-1]
                max_length = max(max_length, current_length)
                print(f"  -> Calculate length: {i} - {stack[-1]} = {current_length}")
                print(f"  -> Max length updated to: {max_length}")
        
        print(f"  -> Stack: {stack}")
    
    print(f"\nFinal result: {max_length}")

def benchmark_longest_valid_parentheses():
    """Benchmark different approaches"""
    import time
    import random
    
    def generate_parentheses_string(length: int, valid_prob: float = 0.7) -> str:
        """Generate random parentheses string"""
        chars = []
        open_count = 0
        
        for _ in range(length):
            if open_count == 0 or (open_count < length // 2 and random.random() < 0.5):
                chars.append('(')
                open_count += 1
            else:
                chars.append(')')
                open_count -= 1
        
        return ''.join(chars)
    
    algorithms = [
        ("Stack Approach", LongestValidParentheses().longestValidParentheses_stack_approach),
        ("DP Approach", LongestValidParentheses().longestValidParentheses_dp_approach),
        ("Two Pass", LongestValidParentheses().longestValidParentheses_two_pass_approach),
        ("Optimized Stack", LongestValidParentheses().longestValidParentheses_optimized_stack),
    ]
    
    string_lengths = [1000, 5000, 10000]
    
    print("\n=== Longest Valid Parentheses Performance Benchmark ===")
    
    for length in string_lengths:
        print(f"\n--- String Length: {length} ---")
        test_strings = [generate_parentheses_string(length) for _ in range(5)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            for test_str in test_strings:
                try:
                    result = alg_func(test_str)
                except:
                    pass  # Skip errors for benchmark
            
            end_time = time.time()
            
            print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = LongestValidParentheses()
    
    edge_cases = [
        ("", 0, "Empty string"),
        ("(", 0, "Single opening"),
        (")", 0, "Single closing"),
        ("()", 2, "Single pair"),
        ("((", 0, "Two opening"),
        ("))", 0, "Two closing"),
        (")(", 0, "Wrong order"),
        ("())", 2, "Extra closing"),
        ("(()", 2, "Extra opening"),
        ("()()()", 6, "All valid"),
        ("((((((", 0, "All opening"),
        ("))))))", 0, "All closing"),
    ]
    
    for s, expected, description in edge_cases:
        result = solver.longestValidParentheses_dp_approach(s)
        status = "✓" if result == expected else "✗"
        print(f"{description:20} | {status} | '{s}' -> {result}")

if __name__ == "__main__":
    test_longest_valid_parentheses()
    demonstrate_dp_approach()
    visualize_stack_approach()
    test_edge_cases()
    benchmark_longest_valid_parentheses()

"""
Longest Valid Parentheses demonstrates advanced stack and DP techniques
for substring validation including two-pass counting, optimized stack
management, and dynamic programming state transitions.
"""
