"""
Problem: Longest Substring with K Unique Characters
URL: https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/description

Problem Statement:
Given a string s and an integer k, return the length of the longest substring of s that contains at most k distinct characters.

Sample Input/Output:
Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.

Input: s = "aa", k = 1
Output: 2
Explanation: The substring is "aa" with length 2.
"""

from typing import List
from collections import defaultdict

class Solution:
    def Longest_Substring_K_Unique_Brute_Force(self, s: str, k: int) -> int:
        """
        Brute Force - Check all substrings
        Time Complexity: O(n²)
        Space Complexity: O(k)
        """
        max_length = 0
        n = len(s)
        
        for i in range(n):
            char_set = set()
            for j in range(i, n):
                char_set.add(s[j])
                if len(char_set) <= k:
                    max_length = max(max_length, j - i + 1)
                else:
                    break
        
        return max_length
    
    def Longest_Substring_K_Unique_Sliding_Window_Optimal(self, s: str, k: int) -> int:
        """
        Sliding Window - Optimal approach
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        if k == 0:
            return 0
        
        char_count = defaultdict(int)
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            char_count[s[right]] += 1
            
            while len(char_count) > k:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length

def Test_Longest_Substring_K_Unique():
    solution = Solution()
    
    test_cases = [
        ("eceba", 2, 3),
        ("aa", 1, 2),
        ("abc", 3, 3),
        ("abcdef", 2, 2)
    ]
    
    for s, k, expected in test_cases:
        result1 = solution.Longest_Substring_K_Unique_Brute_Force(s, k)
        result2 = solution.Longest_Substring_K_Unique_Sliding_Window_Optimal(s, k)
        
        print(f"String: '{s}', k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Sliding Window Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Longest_Substring_K_Unique()
