"""
Problem: Longest Substring without Repeating Characters
URL: https://leetcode.com/problems/longest-substring-without-repeating-characters/

Problem Statement:
Given a string s, find the length of the longest substring without repeating characters.

Sample Input/Output:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
"""

from typing import List

class Solution:
    def Length_Of_Longest_Substring_Brute_Force(self, s: str) -> int:
        """
        Brute Force Approach - Check all substrings
        Time Complexity: O(n³)
        Space Complexity: O(min(m,n)) where m is charset size
        """
        def Has_Unique_Characters(substring: str) -> bool:
            return len(substring) == len(set(substring))
        
        n = len(s)
        max_length = 0
        
        for i in range(n):
            for j in range(i + 1, n + 1):
                if Has_Unique_Characters(s[i:j]):
                    max_length = max(max_length, j - i)
        
        return max_length
    
    def Length_Of_Longest_Substring_Set_Check(self, s: str) -> int:
        """
        Set Check Approach - Use set to check duplicates
        Time Complexity: O(n²)
        Space Complexity: O(min(m,n))
        """
        n = len(s)
        max_length = 0
        
        for i in range(n):
            seen = set()
            for j in range(i, n):
                if s[j] in seen:
                    break
                seen.add(s[j])
                max_length = max(max_length, j - i + 1)
        
        return max_length
    
    def Length_Of_Longest_Substring_Sliding_Window_Optimal(self, s: str) -> int:
        """
        Sliding Window Approach - Two pointers with set
        Time Complexity: O(n)
        Space Complexity: O(min(m,n))
        """
        char_set = set()
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def Length_Of_Longest_Substring_HashMap_Optimized(self, s: str) -> int:
        """
        HashMap Optimized Approach - Store character indices
        Time Complexity: O(n)
        Space Complexity: O(min(m,n))
        """
        char_map = {}
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            if s[right] in char_map and char_map[s[right]] >= left:
                left = char_map[s[right]] + 1
            
            char_map[s[right]] = right
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def Length_Of_Longest_Substring_Array_Optimization(self, s: str) -> int:
        """
        Array Optimization - For ASCII characters
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        char_index = [-1] * 128
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            char = ord(s[right])
            if char_index[char] >= left:
                left = char_index[char] + 1
            
            char_index[char] = right
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def Length_Of_Longest_Substring_Deque_Approach(self, s: str) -> int:
        """
        Deque Approach - Using collections.deque
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        from collections import deque
        
        window = deque()
        max_length = 0
        
        for char in s:
            while char in window:
                window.popleft()
            
            window.append(char)
            max_length = max(max_length, len(window))
        
        return max_length

def Test_Longest_Substring_No_Repeating():
    solution = Solution()
    
    test_cases = [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("", 0),
        ("dvdf", 3),
        ("abcdef", 6)
    ]
    
    for s, expected in test_cases:
        result1 = solution.Length_Of_Longest_Substring_Brute_Force(s)
        result2 = solution.Length_Of_Longest_Substring_Set_Check(s)
        result3 = solution.Length_Of_Longest_Substring_Sliding_Window_Optimal(s)
        result4 = solution.Length_Of_Longest_Substring_HashMap_Optimized(s)
        result5 = solution.Length_Of_Longest_Substring_Array_Optimization(s)
        result6 = solution.Length_Of_Longest_Substring_Deque_Approach(s)
        
        print(f"String: '{s}'")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Set Check: {result2}")
        print(f"Sliding Window Optimal: {result3}")
        print(f"HashMap Optimized: {result4}")
        print(f"Array Optimization: {result5}")
        print(f"Deque Approach: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Longest_Substring_No_Repeating()
