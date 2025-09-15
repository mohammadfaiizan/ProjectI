"""
Problem: Search in Row wise and Column wise Sorted Array
URL: https://leetcode.com/problems/search-a-2d-matrix-ii/description/

Problem Statement:
Write an efficient algorithm that searches for a value target in an m x n integer matrix. 
The matrix has the following properties: Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.

Sample Input/Output:
Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
Output: True
Explanation: 5 is present in the matrix

Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
Output: False
Explanation: 20 is not present in the matrix
"""

from typing import List

class Solution:
    def Search_Matrix_Brute_Force(self, matrix: List[List[int]], target: int) -> bool:
        """
        Brute Force Approach
        Time Complexity: O(m*n)
        Space Complexity: O(1)
        """
        for row in matrix:
            for val in row:
                if val == target:
                    return True
        return False
    
    def Search_Matrix_Row_Wise_Binary_Search(self, matrix: List[List[int]], target: int) -> bool:
        """
        Row-wise Binary Search
        Time Complexity: O(m * log n)
        Space Complexity: O(1)
        """
        def Binary_Search(row: List[int]) -> bool:
            left, right = 0, len(row) - 1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if row[mid] == target:
                    return True
                elif row[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return False
        
        for row in matrix:
            if Binary_Search(row):
                return True
        
        return False
    
    def Search_Matrix_Staircase_Optimal(self, matrix: List[List[int]], target: int) -> bool:
        """
        Staircase Search - Start from top-right corner
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        row, col = 0, len(matrix[0]) - 1
        
        while row < len(matrix) and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False

def Test_Search_2D_Matrix():
    solution = Solution()
    
    matrix = [
        [1,4,7,11,15],
        [2,5,8,12,19],
        [3,6,9,16,22],
        [10,13,14,17,24],
        [18,21,23,26,30]
    ]
    
    test_cases = [
        (matrix, 5, True),
        (matrix, 20, False),
        (matrix, 11, True),
        (matrix, 30, True)
    ]
    
    for matrix_data, target, expected in test_cases:
        result1 = solution.Search_Matrix_Brute_Force(matrix_data, target)
        result2 = solution.Search_Matrix_Row_Wise_Binary_Search(matrix_data, target)
        result3 = solution.Search_Matrix_Staircase_Optimal(matrix_data, target)
        
        print(f"Target: {target}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Row-wise Binary: {result2}")
        print(f"Staircase Optimal: {result3}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Search_2D_Matrix()
