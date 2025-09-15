"""
Problem: Minimum Cost to Cut a Stick
URL: https://leetcode.com/problems/minimum-cost-to-cut-a-stick/description/

Problem Statement:
Given a wooden stick of length n units. The stick is labelled from 0 to n on the axis.
You are given an integer array cuts where cuts[i] denotes a position you should perform a cut at.
You should perform the cuts in order, you can change the order of the cuts as you wish.
The cost of one cut is the length of the stick to be cut, the total cost is the sum of costs of all cuts.
When you cut a stick, the stick gets split into two smaller sticks.
Return the minimum total cost of the cuts.

Sample Input/Output:
Input: n = 7, cuts = [1,3,4,5]
Output: 16
Explanation: Using cuts [1,3,4,5] as in the figure, the minimum total cost is (7) + (2) + (4) + (3) = 16.

Input: n = 9, cuts = [5,6,1,4,2]
Output: 22
Explanation: If you try the given cuts ordering the cost is (9) + (3) + (2) + (1) + (3) = 18.
But if we change the order to [1,2,4,5,6] the cost is (9) + (8) + (6) + (4) + (2) = 29.
The order [5,6,1,4,2] results in the minimum total cost which is (9) + (4) + (6) + (2) + (1) = 22.
"""

from typing import List

class Solution:
    def Min_Cost_Brute_Force(self, n: int, cuts: List[int]) -> int:
        """
        Brute Force - Try all possible cutting orders
        Time Complexity: O(n!)
        Space Complexity: O(n)
        """
        def Calculate_Cost(cuts_order: List[int]) -> int:
            segments = [(0, n)]
            total_cost = 0
            
            for cut in cuts_order:
                for i, (start, end) in enumerate(segments):
                    if start < cut < end:
                        total_cost += end - start
                        segments[i] = (start, cut)
                        segments.insert(i + 1, (cut, end))
                        break
            
            return total_cost
        
        from itertools import permutations
        
        min_cost = float('inf')
        
        for perm in permutations(cuts):
            cost = Calculate_Cost(list(perm))
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    def Min_Cost_Recursive(self, n: int, cuts: List[int]) -> int:
        """
        Recursive - MCM pattern on sorted cuts
        Time Complexity: O(2^m) where m is len(cuts)
        Space Complexity: O(m)
        """
        cuts_with_ends = [0] + sorted(cuts) + [n]
        
        def MCM_Cut(i: int, j: int) -> int:
            if i + 1 >= j:
                return 0
            
            min_cost = float('inf')
            
            for k in range(i + 1, j):
                cost = (MCM_Cut(i, k) + 
                       MCM_Cut(k, j) + 
                       cuts_with_ends[j] - cuts_with_ends[i])
                min_cost = min(min_cost, cost)
            
            return min_cost
        
        return MCM_Cut(0, len(cuts_with_ends) - 1)
    
    def Min_Cost_Memoized(self, n: int, cuts: List[int]) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(m³)
        Space Complexity: O(m²)
        """
        cuts_with_ends = [0] + sorted(cuts) + [n]
        memo = {}
        
        def MCM_Cut_Memo(i: int, j: int) -> int:
            if i + 1 >= j:
                return 0
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            min_cost = float('inf')
            
            for k in range(i + 1, j):
                cost = (MCM_Cut_Memo(i, k) + 
                       MCM_Cut_Memo(k, j) + 
                       cuts_with_ends[j] - cuts_with_ends[i])
                min_cost = min(min_cost, cost)
            
            memo[(i, j)] = min_cost
            return min_cost
        
        return MCM_Cut_Memo(0, len(cuts_with_ends) - 1)
    
    def Min_Cost_Tabulation_Optimal(self, n: int, cuts: List[int]) -> int:
        """
        Tabulation Optimal - Bottom-up DP
        Time Complexity: O(m³)
        Space Complexity: O(m²)
        """
        cuts_with_ends = [0] + sorted(cuts) + [n]
        m = len(cuts_with_ends)
        
        dp = [[0] * m for _ in range(m)]
        
        for length in range(2, m):
            for i in range(m - length):
                j = i + length
                dp[i][j] = float('inf')
                
                for k in range(i + 1, j):
                    cost = dp[i][k] + dp[k][j] + cuts_with_ends[j] - cuts_with_ends[i]
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][m - 1]
    
    def Min_Cost_With_Order(self, n: int, cuts: List[int]) -> tuple:
        """
        With Order - Return cost and optimal cutting order
        Time Complexity: O(m³)
        Space Complexity: O(m²)
        """
        cuts_with_ends = [0] + sorted(cuts) + [n]
        m = len(cuts_with_ends)
        
        dp = [[0] * m for _ in range(m)]
        cut_order = [[[] for _ in range(m)] for _ in range(m)]
        
        for length in range(2, m):
            for i in range(m - length):
                j = i + length
                dp[i][j] = float('inf')
                
                for k in range(i + 1, j):
                    cost = dp[i][k] + dp[k][j] + cuts_with_ends[j] - cuts_with_ends[i]
                    
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        cut_order[i][j] = [cuts_with_ends[k]]
        
        def Get_Cutting_Order(i: int, j: int) -> List[int]:
            if i + 1 >= j:
                return []
            
            order = []
            for cut in cut_order[i][j]:
                k = cuts_with_ends.index(cut)
                order.extend(Get_Cutting_Order(i, k))
                order.append(cut)
                order.extend(Get_Cutting_Order(k, j))
            
            return order
        
        optimal_order = Get_Cutting_Order(0, m - 1)
        filtered_order = [cut for cut in optimal_order if cut in cuts]
        
        return dp[0][m - 1], filtered_order
    
    def Min_Cost_Space_Optimized(self, n: int, cuts: List[int]) -> int:
        """
        Space Optimized - Optimize space using gap method
        Time Complexity: O(m³)
        Space Complexity: O(m²)
        """
        cuts_with_ends = [0] + sorted(cuts) + [n]
        m = len(cuts_with_ends)
        
        dp = [[0] * m for _ in range(m)]
        
        for gap in range(2, m):
            for i in range(m - gap):
                j = i + gap
                dp[i][j] = float('inf')
                
                for k in range(i + 1, j):
                    temp = dp[i][k] + dp[k][j] + cuts_with_ends[j] - cuts_with_ends[i]
                    dp[i][j] = min(dp[i][j], temp)
        
        return dp[0][m - 1]
    
    def Min_Cost_Bottom_Up_Alternative(self, n: int, cuts: List[int]) -> int:
        """
        Bottom Up Alternative - Different iteration order
        Time Complexity: O(m³)
        Space Complexity: O(m²)
        """
        cuts_sorted = sorted([0] + cuts + [n])
        m = len(cuts_sorted)
        
        cost = [[0] * m for _ in range(m)]
        
        for span in range(2, m):
            for left in range(m - span):
                right = left + span
                cost[left][right] = float('inf')
                
                for mid in range(left + 1, right):
                    total = (cost[left][mid] + 
                            cost[mid][right] + 
                            cuts_sorted[right] - cuts_sorted[left])
                    cost[left][right] = min(cost[left][right], total)
        
        return cost[0][m - 1]

def Test_Min_Cost():
    solution = Solution()
    
    test_cases = [
        (7, [1,3,4,5], 16),
        (9, [5,6,1,4,2], 22),
        (20, [1,3,4,5,7,9,10,11], 43),
        (10, [2,4,7], 22),
        (8, [1,2,5], 14)
    ]
    
    methods = [
        ("Recursive", solution.Min_Cost_Recursive),
        ("Memoized", solution.Min_Cost_Memoized),
        ("Tabulation Optimal", solution.Min_Cost_Tabulation_Optimal),
        ("Space Optimized", solution.Min_Cost_Space_Optimized),
        ("Bottom Up Alternative", solution.Min_Cost_Bottom_Up_Alternative)
    ]
    
    for n, cuts, expected in test_cases:
        print(f"Stick length: {n}, Cuts: {cuts}")
        print(f"Expected: {expected}")
        
        if len(cuts) <= 5:
            result_bf = solution.Min_Cost_Brute_Force(n, cuts.copy())
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(n, cuts.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        cost, order = solution.Min_Cost_With_Order(n, cuts.copy())
        print(f"With Order: Cost={cost}, Order={order}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Cost()
