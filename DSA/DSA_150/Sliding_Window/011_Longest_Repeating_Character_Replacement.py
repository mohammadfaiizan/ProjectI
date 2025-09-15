"""
Problem: Longest repeating character replacement
URL: https://leetcode.com/problems/longest-repeating-character-replacement/

Problem Statement:
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. 
You can perform this operation at most k times. Return the length of the longest substring containing the same letter you can get after performing the above operations.

Sample Input/Output:
Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
"""

from typing import List
from collections import defaultdict

class Solution:
    def Character_Replacement_Brute_Force(self, s: str, k: int) -> int:
        """
        Brute Force - Check all substrings
        Time Complexity: O(n² * 26)
        Space Complexity: O(26)
        """
        def Can_Make_Same(substring: str, k: int) -> bool:
            char_count = defaultdict(int)
            for char in substring:
                char_count[char] += 1
            
            max_freq = max(char_count.values())
            return len(substring) - max_freq <= k
        
        max_length = 0
        n = len(s)
        
        for i in range(n):
            for j in range(i + 1, n + 1):
                if Can_Make_Same(s[i:j], k):
                    max_length = max(max_length, j - i)
        
        return max_length
    
    def Character_Replacement_Sliding_Window_Optimal(self, s: str, k: int) -> int:
        """
        Sliding Window - Track character frequencies
        Time Complexity: O(n)
        Space Complexity: O(26)
        """
        char_count = defaultdict(int)
        left = 0
        max_length = 0
        max_freq = 0
        
        for right in range(len(s)):
            char_count[s[right]] += 1
            max_freq = max(max_freq, char_count[s[right]])
            
            if right - left + 1 - max_freq > k:
                char_count[s[left]] -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length

def Test_Character_Replacement():
    solution = Solution()
    
    test_cases = [
        ("ABAB", 2, 4),
        ("AABABBA", 1, 4),
        ("ABCDE", 1, 2),
        ("AAAA", 0, 4)
    ]
    
    for s, k, expected in test_cases:
        result1 = solution.Character_Replacement_Brute_Force(s, k)
        result2 = solution.Character_Replacement_Sliding_Window_Optimal(s, k)
        
        print(f"String: '{s}', k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Sliding Window Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Character_Replacement()
