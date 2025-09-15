"""
Problem: Reverse String
URL: https://leetcode.com/problems/reverse-string/description/

Problem Statement:
Write a function that reverses a string. The input string is given as an array of characters s.
You must do this by modifying the input array in-place with O(1) extra memory.

Sample Input/Output:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]
"""

from typing import List

class Solution:
    def Reverse_String_Built_In(self, s: List[str]) -> None:
        """
        Built-in Reverse - Using Python's reverse method
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        s.reverse()
    
    def Reverse_String_Two_Pointers_Optimal(self, s: List[str]) -> None:
        """
        Two Pointers Optimal - Swap characters from both ends
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(s) - 1
        
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
    
    def Reverse_String_Single_Loop(self, s: List[str]) -> None:
        """
        Single Loop - Use single index and calculate opposite
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(s)
        for i in range(n // 2):
            s[i], s[n - 1 - i] = s[n - 1 - i], s[i]
    
    def Reverse_String_Recursive(self, s: List[str]) -> None:
        """
        Recursive Approach - Recursively swap characters
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        def Reverse_Helper(left: int, right: int) -> None:
            if left >= right:
                return
            
            s[left], s[right] = s[right], s[left]
            Reverse_Helper(left + 1, right - 1)
        
        Reverse_Helper(0, len(s) - 1)
    
    def Reverse_String_Stack_Simulation(self, s: List[str]) -> None:
        """
        Stack Simulation - Simulate stack behavior with two pointers
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        stack_top = len(s) - 1
        
        for i in range(len(s) // 2):
            s[i], s[stack_top] = s[stack_top], s[i]
            stack_top -= 1
    
    def Reverse_String_XOR_Swap(self, s: List[str]) -> None:
        """
        XOR Swap - Use XOR for swapping (works with ASCII values)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(s) - 1
        
        while left < right:
            if s[left] != s[right]:
                s[left] = chr(ord(s[left]) ^ ord(s[right]))
                s[right] = chr(ord(s[left]) ^ ord(s[right]))
                s[left] = chr(ord(s[left]) ^ ord(s[right]))
            left += 1
            right -= 1

def Test_Reverse_String():
    solution = Solution()
    
    test_cases = [
        ["h","e","l","l","o"],
        ["H","a","n","n","a","h"],
        ["a"],
        ["a","b"],
        ["r","a","c","e","c","a","r"]
    ]
    
    methods = [
        ("Built-in", solution.Reverse_String_Built_In),
        ("Two Pointers Optimal", solution.Reverse_String_Two_Pointers_Optimal),
        ("Single Loop", solution.Reverse_String_Single_Loop),
        ("Recursive", solution.Reverse_String_Recursive),
        ("Stack Simulation", solution.Reverse_String_Stack_Simulation),
        ("XOR Swap", solution.Reverse_String_XOR_Swap)
    ]
    
    for s in test_cases:
        original = s.copy()
        expected = s.copy()
        expected.reverse()
        
        print(f"Original: {original}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            test_s = original.copy()
            method(test_s)
            print(f"{method_name}: {test_s}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Reverse_String()
