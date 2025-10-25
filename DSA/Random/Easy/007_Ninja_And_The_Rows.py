"""
Problem: Ninja And The Rows
URL: https://www.naukri.com/code360/problems/ninja-and-the-rows

Problem Statement:
Ninja is given a 2D matrix. He needs to find the row with the maximum sum.
Return the index of the row with the maximum sum (0-indexed).
If multiple rows have the same maximum sum, return the smallest index.

Sample Input/Output:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: 2
Explanation: Row 2 has sum 24, which is maximum

Input: matrix = [[1,1,1],[2,2,2],[1,1,1]]
Output: 1
Explanation: Row 1 has sum 6, which is maximum

Input: matrix = [[5]]
Output: 0
"""

from typing import List

class Solution:
    def Row_Max_Sum_Brute_Force(self, matrix: List[List[int]]) -> int:
        """
        Brute Force Approach - Calculate each row sum
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return 0
        
        max_sum = float('-inf')
        max_row = 0
        
        for i in range(len(matrix)):
            row_sum = 0
            for j in range(len(matrix[i])):
                row_sum += matrix[i][j]
            
            if row_sum > max_sum:
                max_sum = row_sum
                max_row = i
        
        return max_row
    
    def Row_Max_Sum_Built_In(self, matrix: List[List[int]]) -> int:
        """
        Built-in Sum Approach
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return 0
        
        max_sum = float('-inf')
        max_row = 0
        
        for i, row in enumerate(matrix):
            row_sum = sum(row)
            if row_sum > max_sum:
                max_sum = row_sum
                max_row = i
        
        return max_row
    
    def Row_Max_Sum_Enumerate(self, matrix: List[List[int]]) -> int:
        """
        Enumerate Approach - One liner logic
        Time Complexity: O(m * n)
        Space Complexity: O(m)
        """
        if not matrix or not matrix[0]:
            return 0
        
        row_sums = [sum(row) for row in matrix]
        return row_sums.index(max(row_sums))
    
    def Row_Max_Sum_Max_Key(self, matrix: List[List[int]]) -> int:
        """
        Max with Key Approach
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return 0
        
        return max(range(len(matrix)), key=lambda i: sum(matrix[i]))
    
    def Row_Max_Sum_Optimized(self, matrix: List[List[int]]) -> int:
        """
        Optimized Single Pass
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return 0
        
        max_sum = sum(matrix[0])
        max_row = 0
        
        for i in range(1, len(matrix)):
            current_sum = sum(matrix[i])
            if current_sum > max_sum:
                max_sum = current_sum
                max_row = i
        
        return max_row

def Test_Row_Max_Sum():
    solution = Solution()
    
    test_cases = [
        ([[1,2,3],[4,5,6],[7,8,9]], 2),
        ([[1,1,1],[2,2,2],[1,1,1]], 1),
        ([[5]], 0),
        ([[1,2],[3,4],[2,3]], 1),
        ([[10,1,1],[1,1,1],[1,1,10]], 0)
    ]
    
    for matrix, expected in test_cases:
        result1 = solution.Row_Max_Sum_Brute_Force(matrix)
        result2 = solution.Row_Max_Sum_Built_In(matrix)
        result3 = solution.Row_Max_Sum_Enumerate(matrix)
        result4 = solution.Row_Max_Sum_Max_Key(matrix)
        result5 = solution.Row_Max_Sum_Optimized(matrix)
        
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Built-in: {result2}")
        print(f"Enumerate: {result3}")
        print(f"Max Key: {result4}")
        print(f"Optimized: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Row_Max_Sum()

