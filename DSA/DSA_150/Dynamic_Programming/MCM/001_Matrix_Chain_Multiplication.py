"""
Problem: Matrix Chain Multiplication
URL: https://www.geeksforgeeks.org/matrix-chain-multiplication-dp-8/

Problem Statement:
Given a sequence of matrices, find the most efficient way to multiply these matrices together. 
The efficient way is the one that involves the least number of multiplications.
The matrix chain multiplication problem is perhaps the most popular example of dynamic programming.

Sample Input/Output:
Input: p = [1, 2, 3, 4]
Output: 18
Explanation: There are 3 matrices of dimensions 1x2, 2x3, 3x4
The minimum cost is obtained by multiplying in this way: (A1 x A2) x A3 = 1*2*3 + 1*3*4 = 6 + 12 = 18

Input: p = [40, 20, 30, 10, 30]
Output: 26000
Explanation: There are 4 matrices of dimensions 40x20, 20x30, 30x10, 10x30
"""

from typing import List

class Solution:
    def Matrix_Chain_Order_Recursive(self, p: List[int]) -> int:
        """
        Recursive - Try all possible partitions
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        def MCM_Recursive(i: int, j: int) -> int:
            if i >= j:
                return 0
            
            min_cost = float('inf')
            
            for k in range(i, j):
                cost = (MCM_Recursive(i, k) + 
                       MCM_Recursive(k + 1, j) + 
                       p[i - 1] * p[k] * p[j])
                min_cost = min(min_cost, cost)
            
            return min_cost
        
        if len(p) < 2:
            return 0
        
        return MCM_Recursive(1, len(p) - 1)
    
    def Matrix_Chain_Order_Memoized(self, p: List[int]) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(p)
        memo = {}
        
        def MCM_Memo(i: int, j: int) -> int:
            if i >= j:
                return 0
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            min_cost = float('inf')
            
            for k in range(i, j):
                cost = (MCM_Memo(i, k) + 
                       MCM_Memo(k + 1, j) + 
                       p[i - 1] * p[k] * p[j])
                min_cost = min(min_cost, cost)
            
            memo[(i, j)] = min_cost
            return min_cost
        
        if n < 2:
            return 0
        
        return MCM_Memo(1, n - 1)
    
    def Matrix_Chain_Order_Tabulation_Optimal(self, p: List[int]) -> int:
        """
        Tabulation Optimal - Bottom-up DP
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(p)
        if n < 2:
            return 0
        
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for i in range(1, n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j]
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[1][n - 1]
    
    def Matrix_Chain_Order_Space_Optimized(self, p: List[int]) -> int:
        """
        Space Optimized - Reduce space using diagonal processing
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(p)
        if n < 2:
            return 0
        
        dp = [[0] * n for _ in range(n)]
        
        for gap in range(2, n):
            for i in range(1, n - gap + 1):
                j = i + gap - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    temp = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j]
                    dp[i][j] = min(dp[i][j], temp)
        
        return dp[1][n - 1]
    
    def Matrix_Chain_Order_With_Parenthesization(self, p: List[int]) -> tuple:
        """
        With Parenthesization - Return cost and optimal parenthesization
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(p)
        if n < 2:
            return 0, ""
        
        dp = [[0] * n for _ in range(n)]
        bracket = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for i in range(1, n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        bracket[i][j] = k
        
        def Print_Optimal_Parens(s: List[List[int]], i: int, j: int) -> str:
            if i == j:
                return f"M{i}"
            else:
                return (f"({Print_Optimal_Parens(s, i, s[i][j])} x "
                       f"{Print_Optimal_Parens(s, s[i][j] + 1, j)})")
        
        parenthesization = Print_Optimal_Parens(bracket, 1, n - 1)
        return dp[1][n - 1], parenthesization
    
    def Matrix_Chain_Order_All_Ways(self, p: List[int]) -> tuple:
        """
        All Ways - Find all optimal parenthesizations
        Time Complexity: O(n³ + ways)
        Space Complexity: O(n² + ways)
        """
        n = len(p)
        if n < 2:
            return 0, []
        
        dp = [[0] * n for _ in range(n)]
        ways = [[[] for _ in range(n)] for _ in range(n)]
        
        for length in range(2, n):
            for i in range(1, n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j]
                    
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        ways[i][j] = [k]
                    elif cost == dp[i][j]:
                        ways[i][j].append(k)
        
        def Generate_All_Parens(i: int, j: int) -> List[str]:
            if i == j:
                return [f"M{i}"]
            
            all_parens = []
            
            for k in ways[i][j]:
                left_parens = Generate_All_Parens(i, k)
                right_parens = Generate_All_Parens(k + 1, j)
                
                for left in left_parens:
                    for right in right_parens:
                        all_parens.append(f"({left} x {right})")
            
            return all_parens
        
        all_parenthesizations = Generate_All_Parens(1, n - 1)
        return dp[1][n - 1], all_parenthesizations
    
    def Matrix_Chain_Order_Iterative_Bottom_Up(self, p: List[int]) -> int:
        """
        Iterative Bottom Up - Alternative implementation
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(p) - 1
        if n <= 0:
            return 0
        
        m = [[0] * (n + 1) for _ in range(n + 1)]
        
        for l in range(2, n + 1):
            for i in range(1, n - l + 2):
                j = i + l - 1
                m[i][j] = float('inf')
                
                for k in range(i, j):
                    q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                    if q < m[i][j]:
                        m[i][j] = q
        
        return m[1][n]

def Test_Matrix_Chain_Order():
    solution = Solution()
    
    test_cases = [
        ([1, 2, 3, 4], 18),
        ([40, 20, 30, 10, 30], 26000),
        ([1, 2, 3, 4, 5], 38),
        ([5, 4, 6, 2, 7], 158),
        ([2, 3, 4, 5], 54)
    ]
    
    methods = [
        ("Memoized", solution.Matrix_Chain_Order_Memoized),
        ("Tabulation Optimal", solution.Matrix_Chain_Order_Tabulation_Optimal),
        ("Space Optimized", solution.Matrix_Chain_Order_Space_Optimized),
        ("Iterative Bottom Up", solution.Matrix_Chain_Order_Iterative_Bottom_Up)
    ]
    
    for p, expected in test_cases:
        print(f"Dimensions: {p}")
        print(f"Expected: {expected}")
        
        if len(p) <= 6:
            result_rec = solution.Matrix_Chain_Order_Recursive(p.copy())
            print(f"Recursive: {result_rec}")
        
        for method_name, method in methods:
            try:
                result = method(p.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        cost, parenthesization = solution.Matrix_Chain_Order_With_Parenthesization(p.copy())
        print(f"With Parenthesization: Cost={cost}")
        print(f"Optimal Order: {parenthesization}")
        
        if len(p) <= 5:
            cost, all_ways = solution.Matrix_Chain_Order_All_Ways(p.copy())
            print(f"All Ways: Cost={cost}, Count={len(all_ways)}")
            for way in all_ways[:3]:
                print(f"  {way}")
            if len(all_ways) > 3:
                print(f"  ... and {len(all_ways) - 3} more")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Matrix_Chain_Order()
