"""
Problem: Convert To Lower Case
URL: https://leetcode.com/problems/to-lower-case/

Problem Statement:
Given a string s, return the string after replacing every uppercase letter with the same 
lowercase letter.

Sample Input/Output:
Input: s = "Hello"
Output: "hello"

Input: s = "here"
Output: "here"

Input: s = "LOVELY"
Output: "lovely"
"""

from typing import List

class Solution:
    def To_Lower_Case_Built_In(self, s: str) -> str:
        """
        Built-in Method - Using lower()
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return s.lower()
    
    def To_Lower_Case_Manual(self, s: str) -> str:
        """
        Manual Approach - Check ASCII values
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        
        for char in s:
            if 'A' <= char <= 'Z':
                result.append(chr(ord(char) + 32))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def To_Lower_Case_ASCII(self, s: str) -> str:
        """
        ASCII Conversion Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        
        for char in s:
            ascii_val = ord(char)
            if 65 <= ascii_val <= 90:
                result.append(chr(ascii_val + 32))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def To_Lower_Case_List_Comprehension(self, s: str) -> str:
        """
        List Comprehension Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return ''.join([chr(ord(c) + 32) if 'A' <= c <= 'Z' else c for c in s])
    
    def To_Lower_Case_Bit_Manipulation(self, s: str) -> str:
        """
        Bit Manipulation Approach - Set 6th bit
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        
        for char in s:
            if 'A' <= char <= 'Z':
                result.append(chr(ord(char) | 32))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def To_Lower_Case_Map(self, s: str) -> str:
        """
        Map Function Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        def Convert(char: str) -> str:
            if 'A' <= char <= 'Z':
                return chr(ord(char) + 32)
            return char
        
        return ''.join(map(Convert, s))

def Test_To_Lower_Case():
    solution = Solution()
    
    test_cases = [
        ("Hello", "hello"),
        ("here", "here"),
        ("LOVELY", "lovely"),
        ("abc123", "abc123"),
        ("ABC123XYZ", "abc123xyz"),
        ("", "")
    ]
    
    for s, expected in test_cases:
        result1 = solution.To_Lower_Case_Built_In(s)
        result2 = solution.To_Lower_Case_Manual(s)
        result3 = solution.To_Lower_Case_ASCII(s)
        result4 = solution.To_Lower_Case_List_Comprehension(s)
        result5 = solution.To_Lower_Case_Bit_Manipulation(s)
        result6 = solution.To_Lower_Case_Map(s)
        
        print(f"Input: '{s}'")
        print(f"Expected: '{expected}'")
        print(f"Built-in: '{result1}'")
        print(f"Manual: '{result2}'")
        print(f"ASCII: '{result3}'")
        print(f"List Comprehension: '{result4}'")
        print(f"Bit Manipulation: '{result5}'")
        print(f"Map: '{result6}'")
        print("-" * 50)

if __name__ == "__main__":
    Test_To_Lower_Case()

