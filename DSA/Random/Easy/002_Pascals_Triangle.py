"""
Problem: Pascal's Triangle
URL: https://leetcode.com/problems/pascals-triangle/

Problem Statement:
Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it.

Sample Input/Output:
Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

Input: numRows = 1
Output: [[1]]

Input: numRows = 3
Output: [[1],[1,1],[1,2,1]]
"""

from typing import List

class Solution:
    def Pascals_Triangle_Iterative(self, numRows: int) -> List[List[int]]:
        """
        Iterative Approach - Build row by row
        Time Complexity: O(numRows^2)
        Space Complexity: O(1) excluding output
        """
        if numRows == 0:
            return []
        
        result = [[1]]
        
        for i in range(1, numRows):
            prev_row = result[-1]
            new_row = [1]
            
            for j in range(1, i):
                new_row.append(prev_row[j - 1] + prev_row[j])
            
            new_row.append(1)
            result.append(new_row)
        
        return result
    
    def Pascals_Triangle_Dynamic_Programming(self, numRows: int) -> List[List[int]]:
        """
        Dynamic Programming Approach
        Time Complexity: O(numRows^2)
        Space Complexity: O(numRows^2)
        """
        if numRows == 0:
            return []
        
        triangle = [[1]]
        
        for row_num in range(1, numRows):
            row = [1]
            prev_row = triangle[row_num - 1]
            
            for j in range(len(prev_row) - 1):
                row.append(prev_row[j] + prev_row[j + 1])
            
            row.append(1)
            triangle.append(row)
        
        return triangle
    
    def Pascals_Triangle_Mathematical(self, numRows: int) -> List[List[int]]:
        """
        Mathematical Approach using nCr formula
        Time Complexity: O(numRows^2)
        Space Complexity: O(1) excluding output
        """
        def Calculate_NCr(n: int, r: int) -> int:
            if r > n - r:
                r = n - r
            
            result = 1
            for i in range(r):
                result = result * (n - i) // (i + 1)
            
            return result
        
        result = []
        
        for i in range(numRows):
            row = []
            for j in range(i + 1):
                row.append(Calculate_NCr(i, j))
            result.append(row)
        
        return result
    
    def Pascals_Triangle_Optimized(self, numRows: int) -> List[List[int]]:
        """
        Optimized Approach - In-place calculation
        Time Complexity: O(numRows^2)
        Space Complexity: O(1) excluding output
        """
        result = []
        
        for i in range(numRows):
            row = [1] * (i + 1)
            
            for j in range(1, i):
                row[j] = result[i - 1][j - 1] + result[i - 1][j]
            
            result.append(row)
        
        return result
    
    def Pascals_Triangle_Compact(self, numRows: int) -> List[List[int]]:
        """
        Compact Implementation
        Time Complexity: O(numRows^2)
        Space Complexity: O(1) excluding output
        """
        res = [[1]]
        
        for _ in range(numRows - 1):
            res.append([1] + [res[-1][i] + res[-1][i + 1] for i in range(len(res[-1]) - 1)] + [1])
        
        return res if numRows > 0 else []

def Test_Pascals_Triangle():
    solution = Solution()
    
    test_cases = [
        (5, [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]),
        (1, [[1]]),
        (3, [[1],[1,1],[1,2,1]]),
        (4, [[1],[1,1],[1,2,1],[1,3,3,1]]),
        (6, [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1],[1,5,10,10,5,1]])
    ]
    
    for numRows, expected in test_cases:
        result1 = solution.Pascals_Triangle_Iterative(numRows)
        result2 = solution.Pascals_Triangle_Dynamic_Programming(numRows)
        result3 = solution.Pascals_Triangle_Mathematical(numRows)
        result4 = solution.Pascals_Triangle_Optimized(numRows)
        result5 = solution.Pascals_Triangle_Compact(numRows)
        
        print(f"NumRows: {numRows}")
        print(f"Expected: {expected}")
        print(f"Iterative: {result1}")
        print(f"Dynamic Programming: {result2}")
        print(f"Mathematical: {result3}")
        print(f"Optimized: {result4}")
        print(f"Compact: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Pascals_Triangle()

