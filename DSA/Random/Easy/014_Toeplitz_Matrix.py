"""
Problem: Toeplitz Matrix
URL: https://leetcode.com/problems/toeplitz-matrix/

Problem Statement:
Given an m x n matrix, return true if the matrix is Toeplitz. Otherwise, return false.

A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same elements.

Sample Input/Output:
Input: matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
Output: true
Explanation: All diagonals have same elements

Input: matrix = [[1,2],[2,2]]
Output: false

Input: matrix = [[1]]
Output: true
"""

from typing import List

class Solution:
    def Is_Toeplitz_Brute_Force(self, matrix: List[List[int]]) -> bool:
        """
        Brute Force Approach - Check each diagonal
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        m, n = len(matrix), len(matrix[0])
        
        for i in range(m - 1):
            for j in range(n - 1):
                if matrix[i][j] != matrix[i + 1][j + 1]:
                    return False
        
        return True
    
    def Is_Toeplitz_Hash_Map(self, matrix: List[List[int]]) -> bool:
        """
        Hash Map Approach - Store diagonal values
        Time Complexity: O(m * n)
        Space Complexity: O(m + n)
        """
        diagonals = {}
        m, n = len(matrix), len(matrix[0])
        
        for i in range(m):
            for j in range(n):
                diagonal_key = i - j
                
                if diagonal_key in diagonals:
                    if diagonals[diagonal_key] != matrix[i][j]:
                        return False
                else:
                    diagonals[diagonal_key] = matrix[i][j]
        
        return True
    
    def Is_Toeplitz_Optimized(self, matrix: List[List[int]]) -> bool:
        """
        Optimized Approach - Compare with previous row
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        for i in range(1, len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[i][j] != matrix[i - 1][j - 1]:
                    return False
        
        return True
    
    def Is_Toeplitz_Pythonic(self, matrix: List[List[int]]) -> bool:
        """
        Pythonic Approach - Using all() and zip
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        return all(
            matrix[i][j] == matrix[i + 1][j + 1]
            for i in range(len(matrix) - 1)
            for j in range(len(matrix[0]) - 1)
        )
    
    def Is_Toeplitz_Row_Comparison(self, matrix: List[List[int]]) -> bool:
        """
        Row Comparison Approach
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        for i in range(len(matrix) - 1):
            if matrix[i][:-1] != matrix[i + 1][1:]:
                return False
        
        return True

def Test_Is_Toeplitz():
    solution = Solution()
    
    test_cases = [
        ([[1,2,3,4],[5,1,2,3],[9,5,1,2]], True),
        ([[1,2],[2,2]], False),
        ([[1]], True),
        ([[1,2],[2,1]], False),
        ([[18],[66]], True)
    ]
    
    for matrix, expected in test_cases:
        result1 = solution.Is_Toeplitz_Brute_Force([row[:] for row in matrix])
        result2 = solution.Is_Toeplitz_Hash_Map([row[:] for row in matrix])
        result3 = solution.Is_Toeplitz_Optimized([row[:] for row in matrix])
        result4 = solution.Is_Toeplitz_Pythonic([row[:] for row in matrix])
        result5 = solution.Is_Toeplitz_Row_Comparison([row[:] for row in matrix])
        
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Hash Map: {result2}")
        print(f"Optimized: {result3}")
        print(f"Pythonic: {result4}")
        print(f"Row Comparison: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Is_Toeplitz()

