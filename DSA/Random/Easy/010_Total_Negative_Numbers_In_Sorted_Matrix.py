"""
Problem: Total Negative Numbers In A Sorted Matrix
URL: https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/

Problem Statement:
Given a m x n matrix grid which is sorted in non-increasing order both row-wise and 
column-wise, return the number of negative numbers in grid.

Sample Input/Output:
Input: grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]
Output: 8
Explanation: There are 8 negative numbers in the matrix

Input: grid = [[3,2],[1,0]]
Output: 0

Input: grid = [[1,-1],[-1,-1]]
Output: 3
"""

from typing import List

class Solution:
    def Count_Negatives_Brute_Force(self, grid: List[List[int]]) -> int:
        """
        Brute Force Approach - Check each element
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        count = 0
        
        for row in grid:
            for num in row:
                if num < 0:
                    count += 1
        
        return count
    
    def Count_Negatives_Linear_Scan(self, grid: List[List[int]]) -> int:
        """
        Linear Scan from Bottom-Left
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        m, n = len(grid), len(grid[0])
        count = 0
        row, col = m - 1, 0
        
        while row >= 0 and col < n:
            if grid[row][col] < 0:
                count += (n - col)
                row -= 1
            else:
                col += 1
        
        return count
    
    def Count_Negatives_Binary_Search(self, grid: List[List[int]]) -> int:
        """
        Binary Search on Each Row
        Time Complexity: O(m * log n)
        Space Complexity: O(1)
        """
        def First_Negative_Index(row: List[int]) -> int:
            left, right = 0, len(row)
            
            while left < right:
                mid = (left + right) // 2
                if row[mid] < 0:
                    right = mid
                else:
                    left = mid + 1
            
            return left
        
        count = 0
        for row in grid:
            first_neg = First_Negative_Index(row)
            count += len(row) - first_neg
        
        return count
    
    def Count_Negatives_Staircase(self, grid: List[List[int]]) -> int:
        """
        Staircase Search from Top-Right
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        m, n = len(grid), len(grid[0])
        count = 0
        row, col = 0, n - 1
        
        while row < m and col >= 0:
            if grid[row][col] < 0:
                count += (m - row)
                col -= 1
            else:
                row += 1
        
        return count
    
    def Count_Negatives_Pythonic(self, grid: List[List[int]]) -> int:
        """
        Pythonic One-Liner
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        return sum(num < 0 for row in grid for num in row)

def Test_Count_Negatives():
    solution = Solution()
    
    test_cases = [
        ([[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]], 8),
        ([[3,2],[1,0]], 0),
        ([[1,-1],[-1,-1]], 3),
        ([[-1]], 1),
        ([[5,1,0],[-5,-5,-5]], 3)
    ]
    
    for grid, expected in test_cases:
        result1 = solution.Count_Negatives_Brute_Force([row[:] for row in grid])
        result2 = solution.Count_Negatives_Linear_Scan([row[:] for row in grid])
        result3 = solution.Count_Negatives_Binary_Search([row[:] for row in grid])
        result4 = solution.Count_Negatives_Staircase([row[:] for row in grid])
        result5 = solution.Count_Negatives_Pythonic([row[:] for row in grid])
        
        print(f"Grid: {grid}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Linear Scan: {result2}")
        print(f"Binary Search: {result3}")
        print(f"Staircase: {result4}")
        print(f"Pythonic: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Count_Negatives()

