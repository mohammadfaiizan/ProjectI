"""
Problem: Treasure and Jewels
URL: https://leetcode.com/problems/jewels-and-stones/

Problem Statement:
You're given strings jewels representing the types of stones that are jewels, and stones 
representing the stones you have. Each character in stones is a type of stone you have. 
You want to know how many of the stones you have are also jewels.

Letters are case sensitive, so "a" is considered a different type of stone from "A".

Sample Input/Output:
Input: jewels = "aA", stones = "aAAbbbb"
Output: 3

Input: jewels = "z", stones = "ZZ"
Output: 0

Input: jewels = "abc", stones = "aabbccddee"
Output: 6
"""

from typing import List

class Solution:
    def Count_Jewels_Brute_Force(self, jewels: str, stones: str) -> int:
        """
        Brute Force Approach - Check each stone
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        count = 0
        
        for stone in stones:
            for jewel in jewels:
                if stone == jewel:
                    count += 1
                    break
        
        return count
    
    def Count_Jewels_Hash_Set(self, jewels: str, stones: str) -> int:
        """
        Hash Set Approach - Optimal solution
        Time Complexity: O(m + n)
        Space Complexity: O(m)
        """
        jewel_set = set(jewels)
        count = 0
        
        for stone in stones:
            if stone in jewel_set:
                count += 1
        
        return count
    
    def Count_Jewels_Counter(self, jewels: str, stones: str) -> int:
        """
        Counter Approach
        Time Complexity: O(m + n)
        Space Complexity: O(n)
        """
        from collections import Counter
        
        stone_count = Counter(stones)
        total = 0
        
        for jewel in jewels:
            total += stone_count[jewel]
        
        return total
    
    def Count_Jewels_List_Comprehension(self, jewels: str, stones: str) -> int:
        """
        List Comprehension Approach
        Time Complexity: O(m + n)
        Space Complexity: O(m)
        """
        jewel_set = set(jewels)
        return sum(1 for stone in stones if stone in jewel_set)
    
    def Count_Jewels_Pythonic(self, jewels: str, stones: str) -> int:
        """
        Pythonic One-Liner
        Time Complexity: O(m + n)
        Space Complexity: O(m)
        """
        return sum(stone in jewels for stone in stones)

def Test_Count_Jewels():
    solution = Solution()
    
    test_cases = [
        ("aA", "aAAbbbb", 3),
        ("z", "ZZ", 0),
        ("abc", "aabbccddee", 6),
        ("", "abc", 0),
        ("abc", "", 0),
        ("A", "aaAAaa", 2)
    ]
    
    for jewels, stones, expected in test_cases:
        result1 = solution.Count_Jewels_Brute_Force(jewels, stones)
        result2 = solution.Count_Jewels_Hash_Set(jewels, stones)
        result3 = solution.Count_Jewels_Counter(jewels, stones)
        result4 = solution.Count_Jewels_List_Comprehension(jewels, stones)
        result5 = solution.Count_Jewels_Pythonic(jewels, stones)
        
        print(f"Jewels: '{jewels}', Stones: '{stones}'")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Hash Set: {result2}")
        print(f"Counter: {result3}")
        print(f"List Comprehension: {result4}")
        print(f"Pythonic: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Count_Jewels()

