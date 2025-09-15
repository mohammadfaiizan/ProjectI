"""
Problem: Maximal Square
URL: https://leetcode.com/problems/maximal-square/

Problem Statement:
Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

Sample Input/Output:
Input: matrix = [["1","1","1","1","0"],["1","1","1","1","0"],["1","0","0","1","0"],["1","0","0","1","0"],["1","1","1","1","1"]]
Output: 4
Explanation: The largest square has side length 2, so area is 2*2 = 4.

Input: matrix = [["0","1"],["1","0"]]
Output: 1
Explanation: The largest square has side length 1, so area is 1*1 = 1.

Input: matrix = [["0"]]
Output: 0
"""

from typing import List

class Solution:
    def Maximal_Square_Brute_Force(self, matrix: List[List[str]]) -> int:
        """
        Brute Force - Check all possible squares
        Time Complexity: O(m*n*min(m,n)²)
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        max_side = 0
        
        def Is_Valid_Square(row: int, col: int, side: int) -> bool:
            if row + side > m or col + side > n:
                return False
            
            for i in range(row, row + side):
                for j in range(col, col + side):
                    if matrix[i][j] == '0':
                        return False
            
            return True
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    max_possible_side = min(m - i, n - j)
                    
                    for side in range(1, max_possible_side + 1):
                        if Is_Valid_Square(i, j, side):
                            max_side = max(max_side, side)
                        else:
                            break
        
        return max_side * max_side
    
    def Maximal_Square_DP_2D_Optimal(self, matrix: List[List[str]]) -> int:
        """
        DP 2D Optimal - Bottom-up DP with 2D table
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        max_side = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
                    max_side = max(max_side, dp[i][j])
        
        return max_side * max_side
    
    def Maximal_Square_DP_1D_Space_Optimized(self, matrix: List[List[str]]) -> int:
        """
        DP 1D Space Optimized - Use 1D array
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        dp = [0] * n
        max_side = 0
        prev_diagonal = 0
        
        for i in range(m):
            for j in range(n):
                temp = dp[j]
                
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[j] = 1
                    else:
                        dp[j] = min(dp[j], dp[j-1], prev_diagonal) + 1
                    
                    max_side = max(max_side, dp[j])
                else:
                    dp[j] = 0
                
                prev_diagonal = temp
        
        return max_side * max_side
    
    def Maximal_Square_Stack_Based(self, matrix: List[List[str]]) -> int:
        """
        Stack Based - Use stack for histogram approach
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        max_area = 0
        
        def Largest_Square_In_Histogram(heights: List[int]) -> int:
            stack = []
            max_area = 0
            
            for i in range(len(heights)):
                while stack and heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    side = min(h, w)
                    max_area = max(max_area, side * side)
                
                stack.append(i)
            
            while stack:
                h = heights[stack.pop()]
                w = len(heights) if not stack else len(heights) - stack[-1] - 1
                side = min(h, w)
                max_area = max(max_area, side * side)
            
            return max_area
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            max_area = max(max_area, Largest_Square_In_Histogram(heights))
        
        return max_area
    
    def Maximal_Square_Rolling_DP(self, matrix: List[List[str]]) -> int:
        """
        Rolling DP - Use rolling array technique
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        max_side = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if matrix[i-1][j-1] == '1':
                    curr[j] = min(prev[j], curr[j-1], prev[j-1]) + 1
                    max_side = max(max_side, curr[j])
                else:
                    curr[j] = 0
            
            prev, curr = curr, [0] * (n + 1)
        
        return max_side * max_side
    
    def Maximal_Square_Memoized(self, matrix: List[List[str]]) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        memo = {}
        max_side = 0
        
        def DP_Memo(i: int, j: int) -> int:
            nonlocal max_side
            
            if i >= m or j >= n or matrix[i][j] == '0':
                return 0
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i == m - 1 or j == n - 1:
                result = 1 if matrix[i][j] == '1' else 0
            else:
                result = min(DP_Memo(i+1, j), DP_Memo(i, j+1), DP_Memo(i+1, j+1)) + 1
            
            memo[(i, j)] = result
            max_side = max(max_side, result)
            return result
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    DP_Memo(i, j)
        
        return max_side * max_side
    
    def Maximal_Square_With_Position(self, matrix: List[List[str]]) -> tuple:
        """
        With Position - Return area and top-left corner of maximal square
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        if not matrix or not matrix[0]:
            return 0, (-1, -1)
        
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        max_side = 0
        max_position = (-1, -1)
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
                    if dp[i][j] > max_side:
                        max_side = dp[i][j]
                        max_position = (i - max_side + 1, j - max_side + 1)
        
        return max_side * max_side, max_position
    
    def Maximal_Square_BFS(self, matrix: List[List[str]]) -> int:
        """
        BFS - Use breadth-first search to expand squares
        Time Complexity: O(m*n*min(m,n))
        Space Complexity: O(m*n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        from collections import deque
        
        m, n = len(matrix), len(matrix[0])
        max_area = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    queue = deque([(i, j, 1)])
                    visited = set()
                    
                    while queue:
                        row, col, side = queue.popleft()
                        
                        if (row, col, side) in visited:
                            continue
                        
                        visited.add((row, col, side))
                        
                        if row + side <= m and col + side <= n:
                            valid = True
                            
                            for r in range(row, row + side):
                                for c in range(col, col + side):
                                    if matrix[r][c] == '0':
                                        valid = False
                                        break
                                if not valid:
                                    break
                            
                            if valid:
                                max_area = max(max_area, side * side)
                                queue.append((row, col, side + 1))
        
        return max_area

def Test_Maximal_Square():
    solution = Solution()
    
    test_cases = [
        ([["1","1","1","1","0"],["1","1","1","1","0"],["1","0","0","1","0"],["1","0","0","1","0"],["1","1","1","1","1"]], 4),
        ([["0","1"],["1","0"]], 1),
        ([["0"]], 0),
        ([["1"]], 1),
        ([["1","1"],["1","1"]], 4),
        ([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 4),
        ([["0","0","0"],["0","0","0"],["0","0","0"]], 0)
    ]
    
    methods = [
        ("DP 2D Optimal", solution.Maximal_Square_DP_2D_Optimal),
        ("DP 1D Space Optimized", solution.Maximal_Square_DP_1D_Space_Optimized),
        ("Stack Based", solution.Maximal_Square_Stack_Based),
        ("Rolling DP", solution.Maximal_Square_Rolling_DP),
        ("Memoized", solution.Maximal_Square_Memoized)
    ]
    
    for matrix, expected in test_cases:
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        
        if len(matrix) <= 4 and len(matrix[0]) <= 4:
            result_bf = solution.Maximal_Square_Brute_Force([row[:] for row in matrix])
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method([row[:] for row in matrix])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        area, position = solution.Maximal_Square_With_Position([row[:] for row in matrix])
        print(f"With Position: Area={area}, Top-left={position}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Maximal_Square()
