"""
Matrix Optimization Problems
============================

Topics: DP optimization, matrix chain multiplication, advanced optimizations
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Hard
"""

from typing import List, Tuple
import sys

class MatrixOptimization:
    
    # ==========================================
    # 1. MATRIX CHAIN MULTIPLICATION
    # ==========================================
    
    def matrix_chain_order(self, dimensions: List[int]) -> int:
        """Matrix Chain Multiplication - minimum scalar multiplications"""
        n = len(dimensions) - 1
        
        # dp[i][j] = minimum cost to multiply matrices from i to j
        dp = [[0] * n for _ in range(n)]
        
        # l is chain length
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                dp[i][j] = sys.maxsize
                
                for k in range(i, j):
                    cost = (dp[i][k] + dp[k+1][j] + 
                           dimensions[i] * dimensions[k+1] * dimensions[j+1])
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n-1]
    
    def print_optimal_parens(self, s: List[List[int]], i: int, j: int) -> str:
        """Print optimal parenthesization"""
        if i == j:
            return f"M{i}"
        else:
            return f"({self.print_optimal_parens(s, i, s[i][j])}{self.print_optimal_parens(s, s[i][j]+1, j)})"
    
    # ==========================================
    # 2. MATRIX EXPONENTIATION
    # ==========================================
    
    def fibonacci_matrix(self, n: int) -> int:
        """Calculate Fibonacci using matrix exponentiation"""
        if n <= 1:
            return n
        
        def matrix_multiply(A, B):
            return [[A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
                    [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]]
        
        def matrix_power(matrix, power):
            if power == 1:
                return matrix
            
            if power % 2 == 0:
                half = matrix_power(matrix, power // 2)
                return matrix_multiply(half, half)
            else:
                return matrix_multiply(matrix, matrix_power(matrix, power - 1))
        
        base_matrix = [[1, 1], [1, 0]]
        result_matrix = matrix_power(base_matrix, n)
        
        return result_matrix[0][1]
    
    # ==========================================
    # 3. LARGEST RECTANGLE OPTIMIZATIONS
    # ==========================================
    
    def maximal_rectangle_optimized(self, matrix: List[List[str]]) -> int:
        """LC 85: Maximal Rectangle - optimized version"""
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        def largest_rectangle_in_histogram(heights):
            stack = []
            max_area = 0
            
            for i, h in enumerate(heights):
                while stack and heights[stack[-1]] > h:
                    height = heights[stack.pop()]
                    width = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, height * width)
                stack.append(i)
            
            while stack:
                height = heights[stack.pop()]
                width = len(heights) if not stack else len(heights) - stack[-1] - 1
                max_area = max(max_area, height * width)
            
            return max_area
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            max_area = max(max_area, largest_rectangle_in_histogram(heights))
        
        return max_area
    
    # ==========================================
    # 4. MATRIX RANGE QUERIES
    # ==========================================
    
    def range_sum_query_2d(self, matrix: List[List[int]]):
        """LC 304: Range Sum Query 2D - Immutable"""
        class NumMatrix:
            def __init__(self, matrix: List[List[int]]):
                if not matrix or not matrix[0]:
                    return
                
                m, n = len(matrix), len(matrix[0])
                self.prefix_sum = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        self.prefix_sum[i][j] = (matrix[i-1][j-1] + 
                                               self.prefix_sum[i-1][j] + 
                                               self.prefix_sum[i][j-1] - 
                                               self.prefix_sum[i-1][j-1])
            
            def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
                return (self.prefix_sum[row2+1][col2+1] - 
                       self.prefix_sum[row1][col2+1] - 
                       self.prefix_sum[row2+1][col1] + 
                       self.prefix_sum[row1][col1])
        
        return NumMatrix(matrix)
    
    # ==========================================
    # 5. MINIMUM PATH COST OPTIMIZATION
    # ==========================================
    
    def min_falling_path_sum(self, matrix: List[List[int]]) -> int:
        """LC 931: Minimum Falling Path Sum"""
        n = len(matrix)
        
        # Use the matrix itself for DP to save space
        for i in range(1, n):
            for j in range(n):
                min_prev = matrix[i-1][j]
                
                if j > 0:
                    min_prev = min(min_prev, matrix[i-1][j-1])
                if j < n - 1:
                    min_prev = min(min_prev, matrix[i-1][j+1])
                
                matrix[i][j] += min_prev
        
        return min(matrix[n-1])
    
    def min_cost_climbing_stairs_2d(self, cost: List[List[int]]) -> int:
        """2D version of climbing stairs with minimum cost"""
        m, n = len(cost), len(cost[0])
        
        # DP table
        dp = [[float('inf')] * n for _ in range(m)]
        
        # Base cases
        dp[0][0] = cost[0][0]
        if n > 1:
            dp[0][1] = cost[0][1]
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j <= 1:
                    continue
                
                # Can come from top
                if i > 0:
                    dp[i][j] = min(dp[i][j], dp[i-1][j] + cost[i][j])
                
                # Can come from left
                if j > 0:
                    dp[i][j] = min(dp[i][j], dp[i][j-1] + cost[i][j])
                
                # Can come from diagonal
                if i > 0 and j > 0:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-1] + cost[i][j])
        
        return dp[m-1][n-1]
    
    # ==========================================
    # 6. ADVANCED OPTIMIZATION TECHNIQUES
    # ==========================================
    
    def largest_plus_sign(self, n: int, mines: List[List[int]]) -> int:
        """LC 764: Largest Plus Sign"""
        # Create grid with all 1s
        grid = [[1] * n for _ in range(n)]
        
        # Set mines to 0
        for mine in mines:
            grid[mine[0]][mine[1]] = 0
        
        # Arrays to store consecutive 1s in each direction
        left = [[0] * n for _ in range(n)]
        right = [[0] * n for _ in range(n)]
        up = [[0] * n for _ in range(n)]
        down = [[0] * n for _ in range(n)]
        
        # Calculate consecutive 1s from left
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    left[i][j] = 1 if j == 0 else left[i][j-1] + 1
        
        # Calculate consecutive 1s from right
        for i in range(n):
            for j in range(n-1, -1, -1):
                if grid[i][j] == 1:
                    right[i][j] = 1 if j == n-1 else right[i][j+1] + 1
        
        # Calculate consecutive 1s from top
        for j in range(n):
            for i in range(n):
                if grid[i][j] == 1:
                    up[i][j] = 1 if i == 0 else up[i-1][j] + 1
        
        # Calculate consecutive 1s from bottom
        for j in range(n):
            for i in range(n-1, -1, -1):
                if grid[i][j] == 1:
                    down[i][j] = 1 if i == n-1 else down[i+1][j] + 1
        
        # Find maximum plus sign
        max_order = 0
        for i in range(n):
            for j in range(n):
                order = min(left[i][j], right[i][j], up[i][j], down[i][j])
                max_order = max(max_order, order)
        
        return max_order

# Test Examples
def run_examples():
    mo = MatrixOptimization()
    
    print("=== MATRIX OPTIMIZATION PROBLEMS ===\n")
    
    # Matrix Chain Multiplication
    print("1. MATRIX CHAIN MULTIPLICATION:")
    dimensions = [1, 2, 3, 4, 5]
    min_ops = mo.matrix_chain_order(dimensions)
    print(f"Minimum scalar multiplications: {min_ops}")
    
    # Fibonacci using matrix exponentiation
    print("\n2. FIBONACCI WITH MATRIX EXPONENTIATION:")
    n = 10
    fib_result = mo.fibonacci_matrix(n)
    print(f"Fibonacci({n}): {fib_result}")
    
    # Minimum falling path sum
    print("\n3. MINIMUM FALLING PATH SUM:")
    matrix = [
        [2, 1, 3],
        [6, 5, 4],
        [7, 8, 9]
    ]
    min_sum = mo.min_falling_path_sum([row[:] for row in matrix])
    print(f"Minimum falling path sum: {min_sum}")

if __name__ == "__main__":
    run_examples() 