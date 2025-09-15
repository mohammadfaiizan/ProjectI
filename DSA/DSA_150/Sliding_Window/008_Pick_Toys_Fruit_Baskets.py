"""
Problem: Pick Toys (Fruit Into Baskets)
URL: http://leetcode.com/problems/fruit-into-baskets/description

Problem Statement:
You are visiting a farm that has a single row of fruit trees arranged from left to right. 
You have two baskets, and you want to put the maximum number of fruits in your baskets.
You can only have two types of fruits in your baskets.

Sample Input/Output:
Input: fruits = [1,2,1,2,3,1,1]
Output: 5
Explanation: We can collect [2,1,2,3,1] with baskets having types 2 and 1.

Input: fruits = [0,1,2,2]
Output: 3
Explanation: We can collect [1,2,2] with baskets having types 1 and 2.
"""

from typing import List
from collections import defaultdict

class Solution:
    def Total_Fruit_Brute_Force(self, fruits: List[int]) -> int:
        """
        Brute Force - Check all subarrays with at most 2 types
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        max_fruits = 0
        n = len(fruits)
        
        for i in range(n):
            fruit_types = set()
            for j in range(i, n):
                fruit_types.add(fruits[j])
                if len(fruit_types) <= 2:
                    max_fruits = max(max_fruits, j - i + 1)
                else:
                    break
        
        return max_fruits
    
    def Total_Fruit_Sliding_Window_Optimal(self, fruits: List[int]) -> int:
        """
        Sliding Window - At most 2 distinct fruits
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        fruit_count = defaultdict(int)
        left = 0
        max_fruits = 0
        
        for right in range(len(fruits)):
            fruit_count[fruits[right]] += 1
            
            while len(fruit_count) > 2:
                fruit_count[fruits[left]] -= 1
                if fruit_count[fruits[left]] == 0:
                    del fruit_count[fruits[left]]
                left += 1
            
            max_fruits = max(max_fruits, right - left + 1)
        
        return max_fruits

def Test_Total_Fruit():
    solution = Solution()
    
    test_cases = [
        ([1,2,1,2,3,1,1], 5),
        ([0,1,2,2], 3),
        ([1,2,3,2,2], 4),
        ([3,3,3,1,2,1,1,2,3,3,4], 5)
    ]
    
    for fruits, expected in test_cases:
        result1 = solution.Total_Fruit_Brute_Force(fruits.copy())
        result2 = solution.Total_Fruit_Sliding_Window_Optimal(fruits.copy())
        
        print(f"Fruits: {fruits}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Sliding Window Optimal: {result2}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Total_Fruit()
