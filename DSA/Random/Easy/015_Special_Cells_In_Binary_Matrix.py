"""
Problem: Special Cells in Binary Matrix
URL: https://leetcode.com/problems/special-positions-in-a-binary-matrix/

Problem Statement:
Given an m x n binary matrix mat, return the number of special positions in mat.

A position (i, j) is called special if mat[i][j] == 1 and all other elements in row i 
and column j are 0.

Sample Input/Output:
Input: mat = [[1,0,0],[0,0,1],[1,0,0]]
Output: 1
Explanation: Position (1,2) is special

Input: mat = [[1,0,0],[0,1,0],[0,0,1]]
Output: 3

Input: mat = [[0,0,0,1],[1,0,0,0],[0,1,1,0],[0,0,0,0]]
Output: 2
"""

from typing import List

class Solution:
    def Special_Positions_Brute_Force(self, mat: List[List[int]]) -> int:
        """
        Brute Force Approach - Check each cell
        Time Complexity: O(m * n * (m + n))
        Space Complexity: O(1)
        """
        def Is_Special(i: int, j: int) -> bool:
            if mat[i][j] != 1:
                return False
            
            for k in range(len(mat[0])):
                if k != j and mat[i][k] == 1:
                    return False
            
            for k in range(len(mat)):
                if k != i and mat[k][j] == 1:
                    return False
            
            return True
        
        count = 0
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if Is_Special(i, j):
                    count += 1
        
        return count
    
    def Special_Positions_Row_Col_Sum(self, mat: List[List[int]]) -> int:
        """
        Row and Column Sum Approach - Optimal solution
        Time Complexity: O(m * n)
        Space Complexity: O(m + n)
        """
        m, n = len(mat), len(mat[0])
        
        row_sum = [sum(row) for row in mat]
        col_sum = [sum(mat[i][j] for i in range(m)) for j in range(n)]
        
        count = 0
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1 and row_sum[i] == 1 and col_sum[j] == 1:
                    count += 1
        
        return count
    
    def Special_Positions_Precompute(self, mat: List[List[int]]) -> int:
        """
        Precompute Approach
        Time Complexity: O(m * n)
        Space Complexity: O(m + n)
        """
        m, n = len(mat), len(mat[0])
        
        row_counts = [0] * m
        col_counts = [0] * n
        
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    row_counts[i] += 1
                    col_counts[j] += 1
        
        count = 0
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1 and row_counts[i] == 1 and col_counts[j] == 1:
                    count += 1
        
        return count
    
    def Special_Positions_Set(self, mat: List[List[int]]) -> int:
        """
        Set Approach - Track positions
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(mat), len(mat[0])
        
        ones_positions = []
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    ones_positions.append((i, j))
        
        row_sum = [sum(row) for row in mat]
        col_sum = [sum(mat[i][j] for i in range(m)) for j in range(n)]
        
        count = 0
        for i, j in ones_positions:
            if row_sum[i] == 1 and col_sum[j] == 1:
                count += 1
        
        return count
    
    def Special_Positions_List_Comprehension(self, mat: List[List[int]]) -> int:
        """
        List Comprehension Approach
        Time Complexity: O(m * n)
        Space Complexity: O(m + n)
        """
        m, n = len(mat), len(mat[0])
        
        row_sum = [sum(mat[i]) for i in range(m)]
        col_sum = [sum(mat[i][j] for i in range(m)) for j in range(n)]
        
        return sum(
            1 for i in range(m) for j in range(n)
            if mat[i][j] == 1 and row_sum[i] == 1 and col_sum[j] == 1
        )

def Test_Special_Positions():
    solution = Solution()
    
    test_cases = [
        ([[1,0,0],[0,0,1],[1,0,0]], 1),
        ([[1,0,0],[0,1,0],[0,0,1]], 3),
        ([[0,0,0,1],[1,0,0,0],[0,1,1,0],[0,0,0,0]], 2),
        ([[0]], 0),
        ([[1]], 1)
    ]
    
    for mat, expected in test_cases:
        result1 = solution.Special_Positions_Brute_Force([row[:] for row in mat])
        result2 = solution.Special_Positions_Row_Col_Sum([row[:] for row in mat])
        result3 = solution.Special_Positions_Precompute([row[:] for row in mat])
        result4 = solution.Special_Positions_Set([row[:] for row in mat])
        result5 = solution.Special_Positions_List_Comprehension([row[:] for row in mat])
        
        print(f"Matrix: {mat}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Row Col Sum: {result2}")
        print(f"Precompute: {result3}")
        print(f"Set: {result4}")
        print(f"List Comprehension: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Special_Positions()

