"""
Problem: Maximum Nesting Depth Of Two Valid Parentheses Strings
URL: https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/

Problem Statement:
A string is a valid parentheses string if:
- It is the empty string
- It can be written as AB (A concatenated with B), where A and B are valid parentheses strings
- It can be written as (A), where A is a valid parentheses string

You are given a valid parentheses string s. Return the minimum possible maximum nesting 
depth when splitting s into two sequences A and B.

Sample Input/Output:
Input: s = "(()())"
Output: [0,1,1,1,1,0]
Explanation: A = "()()" with depth 1, B = "()" with depth 1

Input: s = "()(())()"
Output: [0,0,0,1,1,0,1,1]

Input: s = "()"
Output: [0,0]
"""

from typing import List

class Solution:
    def Max_Depth_After_Split_Greedy(self, s: str) -> List[int]:
        """
        Greedy Approach - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        depth = 0
        
        for char in s:
            if char == '(':
                depth += 1
                result.append(depth % 2)
            else:
                result.append(depth % 2)
                depth -= 1
        
        return result
    
    def Max_Depth_After_Split_Stack(self, s: str) -> List[int]:
        """
        Stack Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        depth = 0
        
        for char in s:
            if char == '(':
                result.append(depth % 2)
                depth += 1
            else:
                depth -= 1
                result.append(depth % 2)
        
        return result
    
    def Max_Depth_After_Split_Balance(self, s: str) -> List[int]:
        """
        Balance Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        balance = 0
        
        for char in s:
            if char == '(':
                balance += 1
                result.append(balance % 2)
            else:
                result.append(balance % 2)
                balance -= 1
        
        return result
    
    def Max_Depth_After_Split_Enumerate(self, s: str) -> List[int]:
        """
        Enumerate Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        depth = 0
        
        for i, char in enumerate(s):
            if char == '(':
                depth += 1
                result.append(depth % 2)
            else:
                result.append(depth % 2)
                depth -= 1
        
        return result
    
    def Max_Depth_After_Split_List_Comprehension(self, s: str) -> List[int]:
        """
        List Comprehension Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        depth = 0
        result = []
        
        for char in s:
            if char == '(':
                depth += 1
            result.append(depth % 2)
            if char == ')':
                depth -= 1
        
        return result

def Test_Max_Depth_After_Split():
    solution = Solution()
    
    test_cases = [
        "(()())",
        "()(())()",
        "()",
        "((()))",
        "(())()"
    ]
    
    for s in test_cases:
        result1 = solution.Max_Depth_After_Split_Greedy(s)
        result2 = solution.Max_Depth_After_Split_Stack(s)
        result3 = solution.Max_Depth_After_Split_Balance(s)
        result4 = solution.Max_Depth_After_Split_Enumerate(s)
        result5 = solution.Max_Depth_After_Split_List_Comprehension(s)
        
        print(f"String: '{s}'")
        print(f"Greedy: {result1}")
        print(f"Stack: {result2}")
        print(f"Balance: {result3}")
        print(f"Enumerate: {result4}")
        print(f"List Comprehension: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Depth_After_Split()

