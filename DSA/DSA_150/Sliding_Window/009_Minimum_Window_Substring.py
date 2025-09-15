"""
Problem: Minimum Window Substring
URL: https://leetcode.com/problems/minimum-window-substring/description/

Problem Statement:
Given two strings s and t of lengths m and n respectively, return the minimum window substring of s 
such that every character in t (including duplicates) is included in the window.

Sample Input/Output:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
"""

from typing import List
from collections import defaultdict, Counter

class Solution:
    def Min_Window_Brute_Force(self, s: str, t: str) -> str:
        """
        Brute Force Approach - Check all substrings
        Time Complexity: O(n² * m)
        Space Complexity: O(m)
        """
        def Contains_All(window: str, target: str) -> bool:
            target_count = Counter(target)
            window_count = Counter(window)
            
            for char, count in target_count.items():
                if window_count[char] < count:
                    return False
            return True
        
        if len(t) > len(s):
            return ""
        
        min_window = ""
        min_length = float('inf')
        
        for i in range(len(s)):
            for j in range(i + len(t), len(s) + 1):
                window = s[i:j]
                if Contains_All(window, t):
                    if len(window) < min_length:
                        min_length = len(window)
                        min_window = window
                    break
        
        return min_window
    
    def Min_Window_Nested_Loop(self, s: str, t: str) -> str:
        """
        Nested Loop Approach - Expand window from each position
        Time Complexity: O(n² * m)
        Space Complexity: O(m)
        """
        if len(t) > len(s):
            return ""
        
        target_count = Counter(t)
        min_window = ""
        min_length = float('inf')
        
        for i in range(len(s)):
            window_count = defaultdict(int)
            for j in range(i, len(s)):
                window_count[s[j]] += 1
                
                if all(window_count[char] >= count for char, count in target_count.items()):
                    if j - i + 1 < min_length:
                        min_length = j - i + 1
                        min_window = s[i:j+1]
                    break
        
        return min_window
    
    def Min_Window_Sliding_Window_Optimal(self, s: str, t: str) -> str:
        """
        Sliding Window Approach - Optimal solution
        Time Complexity: O(n + m)
        Space Complexity: O(m)
        """
        if len(t) > len(s):
            return ""
        
        target_count = Counter(t)
        window_count = defaultdict(int)
        
        left = 0
        min_length = float('inf')
        min_start = 0
        required = len(target_count)
        formed = 0
        
        for right in range(len(s)):
            char = s[right]
            window_count[char] += 1
            
            if char in target_count and window_count[char] == target_count[char]:
                formed += 1
            
            while left <= right and formed == required:
                if right - left + 1 < min_length:
                    min_length = right - left + 1
                    min_start = left
                
                left_char = s[left]
                window_count[left_char] -= 1
                if left_char in target_count and window_count[left_char] < target_count[left_char]:
                    formed -= 1
                
                left += 1
        
        return "" if min_length == float('inf') else s[min_start:min_start + min_length]
    
    def Min_Window_Two_Pointers_Optimized(self, s: str, t: str) -> str:
        """
        Two Pointers Optimized - Skip irrelevant characters
        Time Complexity: O(n + m)
        Space Complexity: O(m)
        """
        if len(t) > len(s):
            return ""
        
        target_count = Counter(t)
        filtered_s = []
        
        for i, char in enumerate(s):
            if char in target_count:
                filtered_s.append((i, char))
        
        left = 0
        min_length = float('inf')
        min_start = 0
        window_count = defaultdict(int)
        required = len(target_count)
        formed = 0
        
        for right in range(len(filtered_s)):
            char = filtered_s[right][1]
            window_count[char] += 1
            
            if window_count[char] == target_count[char]:
                formed += 1
            
            while left <= right and formed == required:
                start = filtered_s[left][0]
                end = filtered_s[right][0]
                
                if end - start + 1 < min_length:
                    min_length = end - start + 1
                    min_start = start
                
                left_char = filtered_s[left][1]
                window_count[left_char] -= 1
                if window_count[left_char] < target_count[left_char]:
                    formed -= 1
                
                left += 1
        
        return "" if min_length == float('inf') else s[min_start:min_start + min_length]

def Test_Min_Window():
    solution = Solution()
    
    test_cases = [
        ("ADOBECODEBANC", "ABC", "BANC"),
        ("a", "a", "a"),
        ("a", "aa", ""),
        ("ab", "b", "b"),
        ("abc", "ab", "ab")
    ]
    
    for s, t, expected in test_cases:
        result1 = solution.Min_Window_Brute_Force(s, t)
        result2 = solution.Min_Window_Nested_Loop(s, t)
        result3 = solution.Min_Window_Sliding_Window_Optimal(s, t)
        result4 = solution.Min_Window_Two_Pointers_Optimized(s, t)
        
        print(f"String: '{s}', Target: '{t}'")
        print(f"Expected: '{expected}'")
        print(f"Brute Force: '{result1}'")
        print(f"Nested Loop: '{result2}'")
        print(f"Sliding Window Optimal: '{result3}'")
        print(f"Two Pointers Optimized: '{result4}'")
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Window()
