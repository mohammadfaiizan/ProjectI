"""
Problem: Reverse the String
URL: https://leetcode.com/problems/reverse-string/

Problem Statement:
Write a function that reverses a string. The input string is given as an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.

Sample Input/Output:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]

Input: s = ["A"]
Output: ["A"]
"""

from typing import List

class Solution:
    def Reverse_String_Two_Pointer(self, s: List[str]) -> List[str]:
        """
        Two Pointer Approach - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(s) - 1
        
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        
        return s
    
    def Reverse_String_Recursive(self, s: List[str]) -> List[str]:
        """
        Recursive Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        def Reverse_Helper(left: int, right: int):
            if left >= right:
                return
            
            s[left], s[right] = s[right], s[left]
            Reverse_Helper(left + 1, right - 1)
        
        Reverse_Helper(0, len(s) - 1)
        return s
    
    def Reverse_String_Stack(self, s: List[str]) -> List[str]:
        """
        Stack Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = []
        
        for char in s:
            stack.append(char)
        
        for i in range(len(s)):
            s[i] = stack.pop()
        
        return s
    
    def Reverse_String_Pythonic(self, s: List[str]) -> List[str]:
        """
        Pythonic Approach - Using reverse()
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        s.reverse()
        return s
    
    def Reverse_String_Range(self, s: List[str]) -> List[str]:
        """
        Range-based Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(s)
        
        for i in range(n // 2):
            s[i], s[n - 1 - i] = s[n - 1 - i], s[i]
        
        return s
    
    def Reverse_String_Slicing(self, s: List[str]) -> List[str]:
        """
        Slicing Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        s[:] = s[::-1]
        return s

def Test_Reverse_String():
    solution = Solution()
    
    test_cases = [
        (["h","e","l","l","o"], ["o","l","l","e","h"]),
        (["H","a","n","n","a","h"], ["h","a","n","n","a","H"]),
        (["A"], ["A"]),
        (["a","b"], ["b","a"]),
        (["1","2","3","4","5"], ["5","4","3","2","1"])
    ]
    
    for s, expected in test_cases:
        result1 = solution.Reverse_String_Two_Pointer(s.copy())
        result2 = solution.Reverse_String_Recursive(s.copy())
        result3 = solution.Reverse_String_Stack(s.copy())
        result4 = solution.Reverse_String_Pythonic(s.copy())
        result5 = solution.Reverse_String_Range(s.copy())
        result6 = solution.Reverse_String_Slicing(s.copy())
        
        print(f"Input: {s}")
        print(f"Expected: {expected}")
        print(f"Two Pointer: {result1}")
        print(f"Recursive: {result2}")
        print(f"Stack: {result3}")
        print(f"Pythonic: {result4}")
        print(f"Range: {result5}")
        print(f"Slicing: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Reverse_String()

