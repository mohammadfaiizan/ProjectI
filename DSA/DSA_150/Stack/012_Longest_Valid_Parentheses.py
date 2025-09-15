"""
Problem: Longest Valid Parentheses
URL: https://leetcode.com/problems/longest-valid-parentheses/

Problem Statement:
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

Sample Input/Output:
Input: s = "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()".

Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".

Input: s = ""
Output: 0
"""

from typing import List

class Solution:
    def Longest_Valid_Parentheses_Brute_Force(self, s: str) -> int:
        """
        Brute Force Approach - Check all substrings
        Time Complexity: O(n³)
        Space Complexity: O(1)
        """
        def Is_Valid(substring: str) -> bool:
            stack = []
            for char in substring:
                if char == '(':
                    stack.append(char)
                elif stack:
                    stack.pop()
                else:
                    return False
            return len(stack) == 0
        
        max_length = 0
        n = len(s)
        
        for i in range(n):
            for j in range(i + 2, n + 1, 2):
                if Is_Valid(s[i:j]):
                    max_length = max(max_length, j - i)
        
        return max_length
    
    def Longest_Valid_Parentheses_Dynamic_Programming(self, s: str) -> int:
        """
        Dynamic Programming Approach - Build solution bottom up
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not s:
            return 0
        
        n = len(s)
        dp = [0] * n
        max_length = 0
        
        for i in range(1, n):
            if s[i] == ')':
                if s[i - 1] == '(':
                    dp[i] = (dp[i - 2] if i >= 2 else 0) + 2
                elif dp[i - 1] > 0:
                    match_index = i - dp[i - 1] - 1
                    if match_index >= 0 and s[match_index] == '(':
                        dp[i] = dp[i - 1] + 2 + (dp[match_index - 1] if match_index > 0 else 0)
                
                max_length = max(max_length, dp[i])
        
        return max_length
    
    def Longest_Valid_Parentheses_Stack_Optimal(self, s: str) -> int:
        """
        Stack Approach - Track indices of unmatched parentheses
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = [-1]
        max_length = 0
        
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    max_length = max(max_length, i - stack[-1])
        
        return max_length
    
    def Longest_Valid_Parentheses_Two_Pass(self, s: str) -> int:
        """
        Two Pass Approach - Left to right, then right to left
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        def Left_To_Right():
            left = right = max_len = 0
            for char in s:
                if char == '(':
                    left += 1
                else:
                    right += 1
                
                if left == right:
                    max_len = max(max_len, 2 * right)
                elif right > left:
                    left = right = 0
            return max_len
        
        def Right_To_Left():
            left = right = max_len = 0
            for char in reversed(s):
                if char == '(':
                    left += 1
                else:
                    right += 1
                
                if left == right:
                    max_len = max(max_len, 2 * left)
                elif left > right:
                    left = right = 0
            return max_len
        
        return max(Left_To_Right(), Right_To_Left())
    
    def Longest_Valid_Parentheses_Counter_Approach(self, s: str) -> int:
        """
        Counter Approach - Count open and close parentheses
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_length = 0
        open_count = close_count = 0
        
        for char in s:
            if char == '(':
                open_count += 1
            else:
                close_count += 1
            
            if open_count == close_count:
                max_length = max(max_length, 2 * close_count)
            elif close_count > open_count:
                open_count = close_count = 0
        
        open_count = close_count = 0
        for char in reversed(s):
            if char == '(':
                open_count += 1
            else:
                close_count += 1
            
            if open_count == close_count:
                max_length = max(max_length, 2 * open_count)
            elif open_count > close_count:
                open_count = close_count = 0
        
        return max_length
    
    def Longest_Valid_Parentheses_Stack_Indices(self, s: str) -> int:
        """
        Stack with Indices Approach - Store all invalid indices
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = []
        invalid = [False] * len(s)
        
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            else:
                if stack:
                    stack.pop()
                else:
                    invalid[i] = True
        
        for i in stack:
            invalid[i] = True
        
        max_length = current_length = 0
        for is_invalid in invalid:
            if is_invalid:
                current_length = 0
            else:
                current_length += 1
                max_length = max(max_length, current_length)
        
        return max_length

def Test_Longest_Valid_Parentheses():
    solution = Solution()
    
    test_cases = [
        ("(()", 2),
        (")()())", 4),
        ("", 0),
        ("()(()", 2),
        ("(()())", 6),
        ("((()))", 6),
        ("()(())", 6)
    ]
    
    for s, expected in test_cases:
        result1 = solution.Longest_Valid_Parentheses_Brute_Force(s)
        result2 = solution.Longest_Valid_Parentheses_Dynamic_Programming(s)
        result3 = solution.Longest_Valid_Parentheses_Stack_Optimal(s)
        result4 = solution.Longest_Valid_Parentheses_Two_Pass(s)
        result5 = solution.Longest_Valid_Parentheses_Counter_Approach(s)
        result6 = solution.Longest_Valid_Parentheses_Stack_Indices(s)
        
        print(f"String: '{s}'")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Dynamic Programming: {result2}")
        print(f"Stack Optimal: {result3}")
        print(f"Two Pass: {result4}")
        print(f"Counter Approach: {result5}")
        print(f"Stack Indices: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Longest_Valid_Parentheses()
