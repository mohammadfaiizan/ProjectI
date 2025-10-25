"""
Problem: Maximum Points
URL: https://www.naukri.com/code360/problems/maximum-points

Problem Statement:
You are given an array of integers representing points. You can select elements from the 
array but cannot select two adjacent elements. Find the maximum sum of points you can collect.

Sample Input/Output:
Input: points = [1,2,3,1]
Output: 4
Explanation: Select 1 and 3 (indices 0 and 2) = 4

Input: points = [2,7,9,3,1]
Output: 12
Explanation: Select 2, 9, 1 (indices 0, 2, 4) = 12

Input: points = [5,1,1,5]
Output: 10
Explanation: Select 5 and 5 (indices 0 and 3) = 10
"""

from typing import List

class Solution:
    def Maximum_Points_Recursive(self, points: List[int]) -> int:
        """
        Recursive Approach
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        def Rob_Helper(i: int) -> int:
            if i < 0:
                return 0
            if i == 0:
                return points[0]
            
            pick = points[i] + Rob_Helper(i - 2)
            not_pick = Rob_Helper(i - 1)
            
            return max(pick, not_pick)
        
        return Rob_Helper(len(points) - 1)
    
    def Maximum_Points_Memoization(self, points: List[int]) -> int:
        """
        Memoization Approach - Top-down DP
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        memo = {}
        
        def Rob_Helper(i: int) -> int:
            if i < 0:
                return 0
            if i == 0:
                return points[0]
            
            if i in memo:
                return memo[i]
            
            pick = points[i] + Rob_Helper(i - 2)
            not_pick = Rob_Helper(i - 1)
            
            memo[i] = max(pick, not_pick)
            return memo[i]
        
        return Rob_Helper(len(points) - 1)
    
    def Maximum_Points_DP_Tabulation(self, points: List[int]) -> int:
        """
        Dynamic Programming Tabulation - Bottom-up
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not points:
            return 0
        if len(points) == 1:
            return points[0]
        
        n = len(points)
        dp = [0] * n
        
        dp[0] = points[0]
        dp[1] = max(points[0], points[1])
        
        for i in range(2, n):
            dp[i] = max(dp[i - 1], points[i] + dp[i - 2])
        
        return dp[n - 1]
    
    def Maximum_Points_DP_Optimized(self, points: List[int]) -> int:
        """
        Space Optimized DP - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not points:
            return 0
        if len(points) == 1:
            return points[0]
        
        prev2 = points[0]
        prev1 = max(points[0], points[1])
        
        for i in range(2, len(points)):
            current = max(prev1, points[i] + prev2)
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def Maximum_Points_Variables(self, points: List[int]) -> int:
        """
        Two Variables Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not points:
            return 0
        
        rob, not_rob = 0, 0
        
        for point in points:
            new_rob = not_rob + point
            not_rob = max(rob, not_rob)
            rob = new_rob
        
        return max(rob, not_rob)

def Test_Maximum_Points():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,1], 4),
        ([2,7,9,3,1], 12),
        ([5,1,1,5], 10),
        ([1], 1),
        ([1,2], 2),
        ([2,1,1,2], 4)
    ]
    
    for points, expected in test_cases:
        result1 = solution.Maximum_Points_Recursive(points.copy())
        result2 = solution.Maximum_Points_Memoization(points.copy())
        result3 = solution.Maximum_Points_DP_Tabulation(points.copy())
        result4 = solution.Maximum_Points_DP_Optimized(points.copy())
        result5 = solution.Maximum_Points_Variables(points.copy())
        
        print(f"Points: {points}")
        print(f"Expected: {expected}")
        print(f"Recursive: {result1}")
        print(f"Memoization: {result2}")
        print(f"DP Tabulation: {result3}")
        print(f"DP Optimized: {result4}")
        print(f"Variables: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Maximum_Points()

