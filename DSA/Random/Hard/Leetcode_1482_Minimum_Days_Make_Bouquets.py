"""
Problem: Minimum Number of Days to Make m Bouquets
URL: https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/

Problem Statement:
You are given an integer array bloomDay, an integer m and an integer k.

You want to make m bouquets. To make a bouquet, you need to use k adjacent flowers 
from the garden.

The garden consists of n flowers, the ith flower will bloom in the bloomDay[i] and 
then can be used in exactly one bouquet.

Return the minimum number of days you need to wait to be able to make m bouquets from 
the garden. If it is impossible to make m bouquets return -1.

Sample Input/Output:
Input: bloomDay = [1,10,3,10,2], m = 3, k = 1
Output: 3
Explanation: After 3 days: [1,x,3,x,2] -> 3 bouquets of 1 flower each.

Input: bloomDay = [1,10,3,10,2], m = 3, k = 2
Output: -1
Explanation: We need 3 bouquets of 2 adjacent flowers = 6 flowers, but we only have 5.

Input: bloomDay = [7,7,7,7,12,7,7], m = 2, k = 3
Output: 12
"""

from typing import List

class Solution:
    def Min_Days_Brute_Force(self, bloomDay: List[int], m: int, k: int) -> int:
        """
        Brute Force Approach - Try all possible days
        Time Complexity: O(max(bloomDay) * n)
        Space Complexity: O(1)
        """
        n = len(bloomDay)
        if m * k > n:
            return -1
        
        def Can_Make_Bouquets(day: int) -> bool:
            bouquets = 0
            flowers = 0
            
            for bloom in bloomDay:
                if bloom <= day:
                    flowers += 1
                    if flowers == k:
                        bouquets += 1
                        flowers = 0
                else:
                    flowers = 0
            
            return bouquets >= m
        
        min_day = min(bloomDay)
        max_day = max(bloomDay)
        
        for day in range(min_day, max_day + 1):
            if Can_Make_Bouquets(day):
                return day
        
        return -1
    
    def Min_Days_Binary_Search(self, bloomDay: List[int], m: int, k: int) -> int:
        """
        Binary Search Approach
        Time Complexity: O(n * log(max(bloomDay)))
        Space Complexity: O(1)
        """
        n = len(bloomDay)
        if m * k > n:
            return -1
        
        def Can_Make_Bouquets(day: int) -> bool:
            bouquets = 0
            flowers = 0
            
            for bloom in bloomDay:
                if bloom <= day:
                    flowers += 1
                    if flowers == k:
                        bouquets += 1
                        flowers = 0
                else:
                    flowers = 0
            
            return bouquets >= m
        
        left, right = min(bloomDay), max(bloomDay)
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            if Can_Make_Bouquets(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def Min_Days_Binary_Search_Optimal(self, bloomDay: List[int], m: int, k: int) -> int:
        """
        Binary Search Optimal - Most efficient
        Time Complexity: O(n * log(max(bloomDay)))
        Space Complexity: O(1)
        """
        n = len(bloomDay)
        if m * k > n:
            return -1
        
        def Count_Bouquets(day: int) -> int:
            bouquets = 0
            consecutive = 0
            
            for bloom in bloomDay:
                if bloom <= day:
                    consecutive += 1
                    if consecutive == k:
                        bouquets += 1
                        consecutive = 0
                else:
                    consecutive = 0
            
            return bouquets
        
        left, right = min(bloomDay), max(bloomDay)
        
        while left < right:
            mid = (left + right) // 2
            
            if Count_Bouquets(mid) >= m:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def Min_Days_Binary_Search_Greedy(self, bloomDay: List[int], m: int, k: int) -> int:
        """
        Binary Search with Greedy Validation
        Time Complexity: O(n * log(max(bloomDay)))
        Space Complexity: O(1)
        """
        if m * k > len(bloomDay):
            return -1
        
        def Is_Possible(day: int) -> bool:
            bouquet_count = 0
            flower_count = 0
            
            for d in bloomDay:
                if d <= day:
                    flower_count += 1
                    if flower_count >= k:
                        bouquet_count += 1
                        flower_count = 0
                else:
                    flower_count = 0
            
            return bouquet_count >= m
        
        lo, hi = min(bloomDay), max(bloomDay)
        answer = -1
        
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            
            if Is_Possible(mid):
                answer = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        return answer
    
    def Min_Days_Binary_Search_Compact(self, bloomDay: List[int], m: int, k: int) -> int:
        """
        Compact Binary Search Implementation
        Time Complexity: O(n * log(max(bloomDay)))
        Space Complexity: O(1)
        """
        if m * k > len(bloomDay):
            return -1
        
        def Valid(day: int) -> bool:
            b = f = 0
            for d in bloomDay:
                f = f + 1 if d <= day else 0
                if f >= k:
                    b += 1
                    f = 0
            return b >= m
        
        left, right = min(bloomDay), max(bloomDay)
        
        while left < right:
            mid = (left + right) // 2
            if Valid(mid):
                right = mid
            else:
                left = mid + 1
        
        return left

def Test_Min_Days():
    solution = Solution()
    
    test_cases = [
        ([1,10,3,10,2], 3, 1, 3),
        ([1,10,3,10,2], 3, 2, -1),
        ([7,7,7,7,12,7,7], 2, 3, 12),
        ([1,1,1,1], 2, 1, 1),
        ([1,2,3,4,5], 2, 2, 4),
        ([5,37,55,92,22,52,31,62,99,64,92,53,34,84,93,50,28], 8, 2, 93)
    ]
    
    for bloomDay, m, k, expected in test_cases:
        result1 = solution.Min_Days_Brute_Force(bloomDay.copy(), m, k)
        result2 = solution.Min_Days_Binary_Search(bloomDay.copy(), m, k)
        result3 = solution.Min_Days_Binary_Search_Optimal(bloomDay.copy(), m, k)
        result4 = solution.Min_Days_Binary_Search_Greedy(bloomDay.copy(), m, k)
        result5 = solution.Min_Days_Binary_Search_Compact(bloomDay.copy(), m, k)
        
        print(f"BloomDay: {bloomDay}, m: {m}, k: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Binary Search: {result2}")
        print(f"Binary Search Optimal: {result3}")
        print(f"Binary Search Greedy: {result4}")
        print(f"Binary Search Compact: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Days()

