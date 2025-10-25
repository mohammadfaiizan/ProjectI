"""
Problem: Koko Eating Bananas
URL: https://leetcode.com/problems/koko-eating-bananas/

Problem Statement:
Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. 
The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile 
of bananas and eats k bananas from that pile. If the pile has less than k bananas, she 
eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the 
guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.

Sample Input/Output:
Input: piles = [3,6,7,11], h = 8
Output: 4
Explanation: Koko eats at speed 4: 1+2+2+3 = 8 hours.

Input: piles = [30,11,23,4,20], h = 5
Output: 30
Explanation: At speed 30, each pile takes 1 hour: 1+1+1+1+1 = 5 hours.

Input: piles = [30,11,23,4,20], h = 6
Output: 23
"""

from typing import List
import math

class Solution:
    def Min_Eating_Speed_Brute_Force(self, piles: List[int], h: int) -> int:
        """
        Brute Force Approach - Try all speeds
        Time Complexity: O(max(piles) * n)
        Space Complexity: O(1)
        """
        def Can_Finish(speed: int) -> bool:
            hours = 0
            for pile in piles:
                hours += math.ceil(pile / speed)
            return hours <= h
        
        max_pile = max(piles)
        
        for speed in range(1, max_pile + 1):
            if Can_Finish(speed):
                return speed
        
        return max_pile
    
    def Min_Eating_Speed_Binary_Search(self, piles: List[int], h: int) -> int:
        """
        Binary Search Approach
        Time Complexity: O(n * log(max(piles)))
        Space Complexity: O(1)
        """
        def Can_Finish(speed: int) -> bool:
            hours = 0
            for pile in piles:
                hours += (pile + speed - 1) // speed
            return hours <= h
        
        left, right = 1, max(piles)
        result = right
        
        while left <= right:
            mid = (left + right) // 2
            
            if Can_Finish(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def Min_Eating_Speed_Binary_Search_Optimal(self, piles: List[int], h: int) -> int:
        """
        Binary Search Optimal - Most efficient
        Time Complexity: O(n * log(max(piles)))
        Space Complexity: O(1)
        """
        def Hours_Needed(speed: int) -> int:
            hours = 0
            for pile in piles:
                hours += math.ceil(pile / speed)
            return hours
        
        left, right = 1, max(piles)
        
        while left < right:
            mid = (left + right) // 2
            
            if Hours_Needed(mid) <= h:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def Min_Eating_Speed_Binary_Search_Math(self, piles: List[int], h: int) -> int:
        """
        Binary Search with Math Formula
        Time Complexity: O(n * log(max(piles)))
        Space Complexity: O(1)
        """
        def Time_Required(k: int) -> int:
            total = 0
            for p in piles:
                total += (p - 1) // k + 1
            return total
        
        lo, hi = 1, max(piles)
        answer = hi
        
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            
            if Time_Required(mid) <= h:
                answer = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        return answer
    
    def Min_Eating_Speed_Binary_Search_Compact(self, piles: List[int], h: int) -> int:
        """
        Compact Binary Search Implementation
        Time Complexity: O(n * log(max(piles)))
        Space Complexity: O(1)
        """
        def Valid(k: int) -> bool:
            return sum((p - 1) // k + 1 for p in piles) <= h
        
        left, right = 1, max(piles)
        
        while left < right:
            mid = (left + right) // 2
            if Valid(mid):
                right = mid
            else:
                left = mid + 1
        
        return left

def Test_Min_Eating_Speed():
    solution = Solution()
    
    test_cases = [
        ([3,6,7,11], 8, 4),
        ([30,11,23,4,20], 5, 30),
        ([30,11,23,4,20], 6, 23),
        ([1], 1, 1),
        ([1,1,1,1], 4, 1),
        ([10,10,10,10], 5, 10),
        ([312884470], 968709470, 1)
    ]
    
    for piles, h, expected in test_cases:
        result1 = solution.Min_Eating_Speed_Brute_Force(piles.copy(), h) if max(piles) <= 1000 else expected
        result2 = solution.Min_Eating_Speed_Binary_Search(piles.copy(), h)
        result3 = solution.Min_Eating_Speed_Binary_Search_Optimal(piles.copy(), h)
        result4 = solution.Min_Eating_Speed_Binary_Search_Math(piles.copy(), h)
        result5 = solution.Min_Eating_Speed_Binary_Search_Compact(piles.copy(), h)
        
        print(f"Piles: {piles}, Hours: {h}")
        print(f"Expected: {expected}")
        if max(piles) <= 1000:
            print(f"Brute Force: {result1}")
        print(f"Binary Search: {result2}")
        print(f"Binary Search Optimal: {result3}")
        print(f"Binary Search Math: {result4}")
        print(f"Binary Search Compact: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Eating_Speed()

