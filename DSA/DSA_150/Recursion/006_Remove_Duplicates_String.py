"""
Problem: Remove Duplicates from a String
URL: https://leetcode.com/problems/remove-duplicate-letters/description/

Problem Statement:
Given a string s, remove duplicate letters so that every letter appears once and only once.
You must make sure your result is the smallest in lexicographical order among all possible results.

Sample Input/Output:
Input: s = "bcabc"
Output: "abc"
Explanation: Remove duplicate 'b' and 'c' to get lexicographically smallest result

Input: s = "cbacdcbc"
Output: "acdb"
Explanation: Remove duplicates maintaining lexicographical order
"""

from typing import List

class Solution:
    def Remove_Duplicate_Letters_Brute_Force(self, s: str) -> str:
        """
        Brute Force - Remove duplicates while maintaining order
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        result = []
        seen = set()
        
        for char in s:
            if char not in seen:
                result.append(char)
                seen.add(char)
        
        return ''.join(result)
    
    def Remove_Duplicate_Letters_Stack_Optimal(self, s: str) -> str:
        """
        Stack Approach - Optimal lexicographically smallest result
        Time Complexity: O(n)
        Space Complexity: O(1) - limited character set
        """
        count = {}
        for char in s:
            count[char] = count.get(char, 0) + 1
        
        stack = []
        in_stack = set()
        
        for char in s:
            count[char] -= 1
            
            if char in in_stack:
                continue
            
            while (stack and stack[-1] > char and count[stack[-1]] > 0):
                removed = stack.pop()
                in_stack.remove(removed)
            
            stack.append(char)
            in_stack.add(char)
        
        return ''.join(stack)
    
    def Remove_Duplicate_Letters_Recursive(self, s: str) -> str:
        """
        Recursive Approach - Find position and recurse
        Time Complexity: O(n²)
        Space Complexity: O(n) - recursion stack
        """
        if not s:
            return ""
        
        count = {}
        for char in s:
            count[char] = count.get(char, 0) + 1
        
        pos = 0
        for i in range(len(s)):
            if s[i] < s[pos]:
                pos = i
            count[s[i]] -= 1
            if count[s[i]] == 0:
                break
        
        remaining = s[pos + 1:].replace(s[pos], '')
        return s[pos] + self.Remove_Duplicate_Letters_Recursive(remaining)

def Test_Remove_Duplicate_Letters():
    solution = Solution()
    
    test_cases = [
        ("bcabc", "abc"),
        ("cbacdcbc", "acdb"),
        ("abacaba", "abc"),
        ("bbcaac", "bac")
    ]
    
    for s, expected in test_cases:
        result1 = solution.Remove_Duplicate_Letters_Brute_Force(s)
        result2 = solution.Remove_Duplicate_Letters_Stack_Optimal(s)
        result3 = solution.Remove_Duplicate_Letters_Recursive(s)
        
        print(f"String: '{s}'")
        print(f"Expected: '{expected}'")
        print(f"Brute Force: '{result1}'")
        print(f"Stack Optimal: '{result2}'")
        print(f"Recursive: '{result3}'")
        print("-" * 50)

if __name__ == "__main__":
    Test_Remove_Duplicate_Letters()
