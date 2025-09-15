"""
Problem: Paint House II
URL: https://leetcode.com/problems/paint-house-ii/

Problem Statement:
There are a row of n houses, each house can be painted with one of the k colors. 
The cost of painting each house with a certain color is different. 
You have to paint all the houses such that no two adjacent houses have the same color.
The cost of painting each house with a certain color is represented by an n x k cost matrix costs.
For example, costs[0][0] is the cost of painting house 0 with color 0; costs[1][2] is the cost of painting house 1 with color 2, and so on...
Return the minimum cost to paint all houses.

Sample Input/Output:
Input: costs = [[1,5,3],[2,9,4]]
Output: 5
Explanation: Paint house 0 into color 0, paint house 1 into color 2. Minimum cost: 1 + 4 = 5; 
Or paint house 0 into color 2, paint house 1 into color 0. Minimum cost: 3 + 2 = 5.

Input: costs = [[1,3],[2,4]]
Output: 5
Explanation: Paint house 0 into color 0, paint house 1 into color 1. Minimum cost: 1 + 4 = 5; 
Or paint house 0 into color 1, paint house 1 into color 0. Minimum cost: 3 + 2 = 5.
"""

from typing import List, Tuple
import heapq

class Solution:
    def Min_Cost_II_Brute_Force(self, costs: List[List[int]]) -> int:
        """
        Brute Force - Try all possible color combinations
        Time Complexity: O(k^n)
        Space Complexity: O(n)
        """
        if not costs or not costs[0]:
            return 0
        
        n, k = len(costs), len(costs[0])
        
        def Paint_Houses(house: int, prev_color: int) -> int:
            if house >= n:
                return 0
            
            min_cost = float('inf')
            
            for color in range(k):
                if color != prev_color:
                    cost = costs[house][color] + Paint_Houses(house + 1, color)
                    min_cost = min(min_cost, cost)
            
            return min_cost
        
        return Paint_Houses(0, -1)
    
    def Min_Cost_II_Memoized(self, costs: List[List[int]]) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(n * k²)
        Space Complexity: O(n * k)
        """
        if not costs or not costs[0]:
            return 0
        
        n, k = len(costs), len(costs[0])
        memo = {}
        
        def Paint_Memo(house: int, prev_color: int) -> int:
            if house >= n:
                return 0
            
            if (house, prev_color) in memo:
                return memo[(house, prev_color)]
            
            min_cost = float('inf')
            
            for color in range(k):
                if color != prev_color:
                    cost = costs[house][color] + Paint_Memo(house + 1, color)
                    min_cost = min(min_cost, cost)
            
            memo[(house, prev_color)] = min_cost
            return min_cost
        
        return Paint_Memo(0, -1)
    
    def Min_Cost_II_DP_2D(self, costs: List[List[int]]) -> int:
        """
        DP 2D - Bottom-up DP with 2D table
        Time Complexity: O(n * k²)
        Space Complexity: O(n * k)
        """
        if not costs or not costs[0]:
            return 0
        
        n, k = len(costs), len(costs[0])
        
        dp = [[0] * k for _ in range(n)]
        
        for j in range(k):
            dp[0][j] = costs[0][j]
        
        for i in range(1, n):
            for j in range(k):
                dp[i][j] = float('inf')
                
                for prev_j in range(k):
                    if prev_j != j:
                        dp[i][j] = min(dp[i][j], dp[i-1][prev_j] + costs[i][j])
        
        return min(dp[n-1])
    
    def Min_Cost_II_Space_Optimized_Optimal(self, costs: List[List[int]]) -> int:
        """
        Space Optimized Optimal - Track min and second min
        Time Complexity: O(n * k)
        Space Complexity: O(1)
        """
        if not costs or not costs[0]:
            return 0
        
        n, k = len(costs), len(costs[0])
        
        min1 = min2 = 0
        min1_idx = -1
        
        for i in range(n):
            new_min1 = new_min2 = float('inf')
            new_min1_idx = -1
            
            for j in range(k):
                if j == min1_idx:
                    cost = min2 + costs[i][j]
                else:
                    cost = min1 + costs[i][j]
                
                if cost < new_min1:
                    new_min2 = new_min1
                    new_min1 = cost
                    new_min1_idx = j
                elif cost < new_min2:
                    new_min2 = cost
            
            min1, min2, min1_idx = new_min1, new_min2, new_min1_idx
        
        return min1
    
    def Min_Cost_II_Rolling_Array(self, costs: List[List[int]]) -> int:
        """
        Rolling Array - Use two arrays alternating
        Time Complexity: O(n * k²)
        Space Complexity: O(k)
        """
        if not costs or not costs[0]:
            return 0
        
        n, k = len(costs), len(costs[0])
        
        prev = costs[0][:]
        curr = [0] * k
        
        for i in range(1, n):
            for j in range(k):
                curr[j] = float('inf')
                
                for prev_j in range(k):
                    if prev_j != j:
                        curr[j] = min(curr[j], prev[prev_j] + costs[i][j])
            
            prev, curr = curr, prev
        
        return min(prev)
    
    def Min_Cost_II_With_Color_Sequence(self, costs: List[List[int]]) -> Tuple[int, List[int]]:
        """
        With Color Sequence - Return min cost and color sequence
        Time Complexity: O(n * k²)
        Space Complexity: O(n * k)
        """
        if not costs or not costs[0]:
            return 0, []
        
        n, k = len(costs), len(costs[0])
        
        dp = [[float('inf')] * k for _ in range(n)]
        parent = [[-1] * k for _ in range(n)]
        
        for j in range(k):
            dp[0][j] = costs[0][j]
        
        for i in range(1, n):
            for j in range(k):
                for prev_j in range(k):
                    if prev_j != j:
                        cost = dp[i-1][prev_j] + costs[i][j]
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            parent[i][j] = prev_j
        
        min_cost = min(dp[n-1])
        last_color = dp[n-1].index(min_cost)
        
        color_sequence = []
        current_color = last_color
        
        for i in range(n - 1, -1, -1):
            color_sequence.append(current_color)
            if i > 0:
                current_color = parent[i][current_color]
        
        return min_cost, color_sequence[::-1]
    
    def Min_Cost_II_Heap_Optimization(self, costs: List[List[int]]) -> int:
        """
        Heap Optimization - Use heap to find minimum efficiently
        Time Complexity: O(n * k * log k)
        Space Complexity: O(k)
        """
        if not costs or not costs[0]:
            return 0
        
        n, k = len(costs), len(costs[0])
        
        prev_costs = [(costs[0][j], j) for j in range(k)]
        heapq.heapify(prev_costs)
        
        for i in range(1, n):
            curr_costs = []
            
            for j in range(k):
                min_prev_cost = float('inf')
                
                temp_heap = prev_costs[:]
                while temp_heap:
                    cost, color = heapq.heappop(temp_heap)
                    if color != j:
                        min_prev_cost = cost
                        break
                
                curr_costs.append((min_prev_cost + costs[i][j], j))
            
            prev_costs = curr_costs
            heapq.heapify(prev_costs)
        
        return min(cost for cost, _ in prev_costs)
    
    def Min_Cost_II_Iterative_Min_Finding(self, costs: List[List[int]]) -> int:
        """
        Iterative Min Finding - Alternative approach to find minimums
        Time Complexity: O(n * k)
        Space Complexity: O(1)
        """
        if not costs or not costs[0]:
            return 0
        
        n, k = len(costs), len(costs[0])
        
        def Find_Two_Minimums(arr: List[int]) -> Tuple[int, int, int]:
            min1 = min2 = float('inf')
            min1_idx = -1
            
            for i, val in enumerate(arr):
                if val < min1:
                    min2 = min1
                    min1 = val
                    min1_idx = i
                elif val < min2:
                    min2 = val
            
            return min1, min2, min1_idx
        
        prev = costs[0][:]
        
        for i in range(1, n):
            min1, min2, min1_idx = Find_Two_Minimums(prev)
            curr = [0] * k
            
            for j in range(k):
                if j == min1_idx:
                    curr[j] = min2 + costs[i][j]
                else:
                    curr[j] = min1 + costs[i][j]
            
            prev = curr
        
        return min(prev)

def Test_Min_Cost_II():
    solution = Solution()
    
    test_cases = [
        ([[1,5,3],[2,9,4]], 5),
        ([[1,3],[2,4]], 5),
        ([[1,2,3],[1,4,6]], 3),
        ([[3,5,3],[6,17,6],[7,13,18],[9,10,18]], 16),
        ([[2,1,3,4],[5,8,4,3],[6,1,2,4],[1,3,5,2]], 8)
    ]
    
    methods = [
        ("Memoized", solution.Min_Cost_II_Memoized),
        ("DP 2D", solution.Min_Cost_II_DP_2D),
        ("Space Optimized Optimal", solution.Min_Cost_II_Space_Optimized_Optimal),
        ("Rolling Array", solution.Min_Cost_II_Rolling_Array),
        ("Heap Optimization", solution.Min_Cost_II_Heap_Optimization),
        ("Iterative Min Finding", solution.Min_Cost_II_Iterative_Min_Finding)
    ]
    
    for costs, expected in test_cases:
        print(f"Costs: {costs}")
        print(f"Expected: {expected}")
        
        if len(costs) <= 4 and len(costs[0]) <= 4:
            result_bf = solution.Min_Cost_II_Brute_Force([row[:] for row in costs])
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method([row[:] for row in costs])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        min_cost, color_sequence = solution.Min_Cost_II_With_Color_Sequence([row[:] for row in costs])
        print(f"With Colors: Cost={min_cost}, Sequence={color_sequence}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Cost_II()
