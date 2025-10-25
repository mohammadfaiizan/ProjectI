"""
Problem: Split String
URL: https://leetcode.com/problems/split-a-string-in-balanced-strings/

Problem Statement:
Balanced strings are those that have an equal quantity of 'L' and 'R' characters.

Given a balanced string s, split it into some number of substrings such that:
- Each substring is balanced.

Return the maximum amount of balanced strings you can obtain.

Sample Input/Output:
Input: s = "RLRRLLRLRL"
Output: 4
Explanation: s can be split into "RL", "RRLL", "RL", "RL", each substring contains equal L and R.

Input: s = "RLRRRLLRLL"
Output: 2

Input: s = "LLLLRRRR"
Output: 1
"""

from typing import List

class Solution:
    def Balanced_String_Split_Greedy(self, s: str) -> int:
        """
        Greedy Approach - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        balance = 0
        count = 0
        
        for char in s:
            if char == 'L':
                balance += 1
            else:
                balance -= 1
            
            if balance == 0:
                count += 1
        
        return count
    
    def Balanced_String_Split_Counter(self, s: str) -> int:
        """
        Counter Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        l_count = 0
        r_count = 0
        result = 0
        
        for char in s:
            if char == 'L':
                l_count += 1
            else:
                r_count += 1
            
            if l_count == r_count:
                result += 1
                l_count = 0
                r_count = 0
        
        return result
    
    def Balanced_String_Split_Stack(self, s: str) -> int:
        """
        Stack-like Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        balance = 0
        splits = 0
        
        for char in s:
            balance += 1 if char == 'R' else -1
            
            if balance == 0:
                splits += 1
        
        return splits
    
    def Balanced_String_Split_Enumeration(self, s: str) -> int:
        """
        Enumeration Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        count = 0
        balance = 0
        
        for i, char in enumerate(s):
            balance += (1 if char == 'R' else -1)
            count += (balance == 0)
        
        return count
    
    def Balanced_String_Split_Reduce(self, s: str) -> int:
        """
        Reduce Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        from functools import reduce
        
        def Process(acc, char):
            balance, count = acc
            balance += (1 if char == 'R' else -1)
            count += (1 if balance == 0 else 0)
            return (balance, count)
        
        _, result = reduce(Process, s, (0, 0))
        return result

def Test_Balanced_String_Split():
    solution = Solution()
    
    test_cases = [
        ("RLRRLLRLRL", 4),
        ("RLRRRLLRLL", 2),
        ("LLLLRRRR", 1),
        ("RL", 1),
        ("RLRL", 2)
    ]
    
    for s, expected in test_cases:
        result1 = solution.Balanced_String_Split_Greedy(s)
        result2 = solution.Balanced_String_Split_Counter(s)
        result3 = solution.Balanced_String_Split_Stack(s)
        result4 = solution.Balanced_String_Split_Enumeration(s)
        result5 = solution.Balanced_String_Split_Reduce(s)
        
        print(f"String: '{s}'")
        print(f"Expected: {expected}")
        print(f"Greedy: {result1}")
        print(f"Counter: {result2}")
        print(f"Stack: {result3}")
        print(f"Enumeration: {result4}")
        print(f"Reduce: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Balanced_String_Split()

