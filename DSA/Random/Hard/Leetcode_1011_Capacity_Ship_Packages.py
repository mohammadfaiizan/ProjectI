"""
Problem: Capacity To Ship Packages Within D Days
URL: https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/

Problem Statement:
A conveyor belt has packages that must be shipped from one port to another within days days.

The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship 
with packages on the conveyor belt (in the order given by weights). We may not load more 
weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the 
conveyor belt being shipped within days days.

Sample Input/Output:
Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
Explanation: Ship with capacity 15 can ship: [1,2,3,4,5], [6,7], [8], [9], [10] in 5 days.

Input: weights = [3,2,2,4,1,4], days = 3
Output: 6
Explanation: Ship with capacity 6: [3,2], [2,4], [1,4] in 3 days.

Input: weights = [1,2,3,1,1], days = 4
Output: 3
"""

from typing import List

class Solution:
    def Ship_Within_Days_Brute_Force(self, weights: List[int], days: int) -> int:
        """
        Brute Force Approach - Try all capacities
        Time Complexity: O(n * sum(weights))
        Space Complexity: O(1)
        """
        def Can_Ship(capacity: int) -> bool:
            days_needed = 1
            current_weight = 0
            
            for weight in weights:
                if current_weight + weight > capacity:
                    days_needed += 1
                    current_weight = weight
                else:
                    current_weight += weight
            
            return days_needed <= days
        
        max_weight = max(weights)
        total_weight = sum(weights)
        
        for capacity in range(max_weight, total_weight + 1):
            if Can_Ship(capacity):
                return capacity
        
        return total_weight
    
    def Ship_Within_Days_Binary_Search(self, weights: List[int], days: int) -> int:
        """
        Binary Search Approach
        Time Complexity: O(n * log(sum(weights)))
        Space Complexity: O(1)
        """
        def Can_Ship(capacity: int) -> bool:
            days_needed = 1
            current_load = 0
            
            for weight in weights:
                if current_load + weight > capacity:
                    days_needed += 1
                    current_load = weight
                    if days_needed > days:
                        return False
                else:
                    current_load += weight
            
            return True
        
        left, right = max(weights), sum(weights)
        result = right
        
        while left <= right:
            mid = (left + right) // 2
            
            if Can_Ship(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def Ship_Within_Days_Binary_Search_Optimal(self, weights: List[int], days: int) -> int:
        """
        Binary Search Optimal - Most efficient
        Time Complexity: O(n * log(sum(weights)))
        Space Complexity: O(1)
        """
        def Days_Needed(capacity: int) -> int:
            day_count = 1
            current = 0
            
            for weight in weights:
                if current + weight > capacity:
                    day_count += 1
                    current = weight
                else:
                    current += weight
            
            return day_count
        
        left, right = max(weights), sum(weights)
        
        while left < right:
            mid = (left + right) // 2
            
            if Days_Needed(mid) <= days:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def Ship_Within_Days_Binary_Search_Greedy(self, weights: List[int], days: int) -> int:
        """
        Binary Search with Greedy Validation
        Time Complexity: O(n * log(sum(weights)))
        Space Complexity: O(1)
        """
        def Is_Feasible(capacity: int) -> bool:
            required_days = 1
            load = 0
            
            for w in weights:
                load += w
                if load > capacity:
                    required_days += 1
                    load = w
            
            return required_days <= days
        
        lo, hi = max(weights), sum(weights)
        answer = hi
        
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            
            if Is_Feasible(mid):
                answer = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        return answer
    
    def Ship_Within_Days_Binary_Search_Compact(self, weights: List[int], days: int) -> int:
        """
        Compact Binary Search Implementation
        Time Complexity: O(n * log(sum(weights)))
        Space Complexity: O(1)
        """
        def Valid(cap: int) -> bool:
            d, curr = 1, 0
            for w in weights:
                curr += w
                if curr > cap:
                    d += 1
                    curr = w
            return d <= days
        
        left, right = max(weights), sum(weights)
        
        while left < right:
            mid = (left + right) // 2
            if Valid(mid):
                right = mid
            else:
                left = mid + 1
        
        return left

def Test_Ship_Within_Days():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,4,5,6,7,8,9,10], 5, 15),
        ([3,2,2,4,1,4], 3, 6),
        ([1,2,3,1,1], 4, 3),
        ([10], 1, 10),
        ([1,1,1,1], 1, 4),
        ([1,2,3,4,5], 1, 15)
    ]
    
    for weights, days, expected in test_cases:
        result1 = solution.Ship_Within_Days_Brute_Force(weights.copy(), days)
        result2 = solution.Ship_Within_Days_Binary_Search(weights.copy(), days)
        result3 = solution.Ship_Within_Days_Binary_Search_Optimal(weights.copy(), days)
        result4 = solution.Ship_Within_Days_Binary_Search_Greedy(weights.copy(), days)
        result5 = solution.Ship_Within_Days_Binary_Search_Compact(weights.copy(), days)
        
        print(f"Weights: {weights}, Days: {days}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Binary Search: {result2}")
        print(f"Binary Search Optimal: {result3}")
        print(f"Binary Search Greedy: {result4}")
        print(f"Binary Search Compact: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Ship_Within_Days()

