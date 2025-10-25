"""
Problem: Buildings Projection
URL: https://www.naukri.com/code360/problems/buildings-projection

Problem Statement:
Given a 2D grid representing building heights, find how many buildings can be seen from 
the top, right, bottom, and left sides. A building is visible from a side if there's no 
taller building blocking it from that direction.

Return the total count of visible buildings from all four sides.

Sample Input/Output:
Input: grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
Output: 35

Input: grid = [[1,2,3],[4,5,6],[7,8,9]]
Output: 16

Input: grid = [[1]]
Output: 1
"""

from typing import List

class Solution:
    def Visible_Buildings_Brute_Force(self, grid: List[List[int]]) -> int:
        """
        Brute Force Approach - Check from all sides
        Time Complexity: O(m * n * (m + n))
        Space Complexity: O(1)
        """
        def Count_From_Side(grid: List[List[int]]) -> int:
            count = 0
            for row in grid:
                max_height = 0
                for height in row:
                    if height > max_height:
                        count += 1
                        max_height = height
            return count
        
        visible = 0
        
        visible += Count_From_Side(grid)
        
        visible += Count_From_Side([row[::-1] for row in grid])
        
        transposed = [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]
        visible += Count_From_Side(transposed)
        
        visible += Count_From_Side([row[::-1] for row in transposed])
        
        return visible
    
    def Visible_Buildings_Optimized(self, grid: List[List[int]]) -> int:
        """
        Optimized Approach
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        m, n = len(grid), len(grid[0])
        count = 0
        
        for i in range(m):
            max_left = 0
            max_right = 0
            for j in range(n):
                if grid[i][j] > max_left:
                    count += 1
                    max_left = grid[i][j]
                
                if grid[i][n - 1 - j] > max_right:
                    count += 1
                    max_right = grid[i][n - 1 - j]
        
        for j in range(n):
            max_top = 0
            max_bottom = 0
            for i in range(m):
                if grid[i][j] > max_top:
                    count += 1
                    max_top = grid[i][j]
                
                if grid[m - 1 - i][j] > max_bottom:
                    count += 1
                    max_bottom = grid[m - 1 - i][j]
        
        return count
    
    def Visible_Buildings_Four_Directions(self, grid: List[List[int]]) -> int:
        """
        Four Directions Approach - Clean implementation
        Time Complexity: O(m * n)
        Space Complexity: O(1)
        """
        m, n = len(grid), len(grid[0])
        visible = 0
        
        for i in range(m):
            max_h = 0
            for j in range(n):
                if grid[i][j] > max_h:
                    visible += 1
                    max_h = grid[i][j]
        
        for i in range(m):
            max_h = 0
            for j in range(n - 1, -1, -1):
                if grid[i][j] > max_h:
                    visible += 1
                    max_h = grid[i][j]
        
        for j in range(n):
            max_h = 0
            for i in range(m):
                if grid[i][j] > max_h:
                    visible += 1
                    max_h = grid[i][j]
        
        for j in range(n):
            max_h = 0
            for i in range(m - 1, -1, -1):
                if grid[i][j] > max_h:
                    visible += 1
                    max_h = grid[i][j]
        
        return visible
    
    def Visible_Buildings_Set_Tracking(self, grid: List[List[int]]) -> int:
        """
        Set Tracking Approach - Track unique visible buildings
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(grid), len(grid[0])
        visible_set = set()
        
        for i in range(m):
            max_h = 0
            for j in range(n):
                if grid[i][j] > max_h:
                    visible_set.add((i, j))
                    max_h = grid[i][j]
        
        for i in range(m):
            max_h = 0
            for j in range(n - 1, -1, -1):
                if grid[i][j] > max_h:
                    visible_set.add((i, j))
                    max_h = grid[i][j]
        
        for j in range(n):
            max_h = 0
            for i in range(m):
                if grid[i][j] > max_h:
                    visible_set.add((i, j))
                    max_h = grid[i][j]
        
        for j in range(n):
            max_h = 0
            for i in range(m - 1, -1, -1):
                if grid[i][j] > max_h:
                    visible_set.add((i, j))
                    max_h = grid[i][j]
        
        return len(visible_set)
    
    def Visible_Buildings_Matrix_Tracking(self, grid: List[List[int]]) -> int:
        """
        Matrix Tracking Approach
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(grid), len(grid[0])
        visible = [[False] * n for _ in range(m)]
        
        directions = [
            (range(m), range(n)),
            (range(m), range(n - 1, -1, -1)),
            (range(m), range(n)),
            (range(m - 1, -1, -1), range(n))
        ]
        
        for i in range(m):
            max_h = 0
            for j in range(n):
                if grid[i][j] > max_h:
                    visible[i][j] = True
                    max_h = grid[i][j]
        
        for i in range(m):
            max_h = 0
            for j in range(n - 1, -1, -1):
                if grid[i][j] > max_h:
                    visible[i][j] = True
                    max_h = grid[i][j]
        
        for j in range(n):
            max_h = 0
            for i in range(m):
                if grid[i][j] > max_h:
                    visible[i][j] = True
                    max_h = grid[i][j]
        
        for j in range(n):
            max_h = 0
            for i in range(m - 1, -1, -1):
                if grid[i][j] > max_h:
                    visible[i][j] = True
                    max_h = grid[i][j]
        
        return sum(sum(row) for row in visible)

def Test_Visible_Buildings():
    solution = Solution()
    
    test_cases = [
        ([[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]], 35),
        ([[1,2,3],[4,5,6],[7,8,9]], 16),
        ([[1]], 1),
        ([[1,2],[3,4]], 4)
    ]
    
    for grid, expected in test_cases:
        result1 = solution.Visible_Buildings_Brute_Force([row[:] for row in grid])
        result2 = solution.Visible_Buildings_Optimized([row[:] for row in grid])
        result3 = solution.Visible_Buildings_Four_Directions([row[:] for row in grid])
        result4 = solution.Visible_Buildings_Set_Tracking([row[:] for row in grid])
        result5 = solution.Visible_Buildings_Matrix_Tracking([row[:] for row in grid])
        
        print(f"Grid: {grid}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Optimized: {result2}")
        print(f"Four Directions: {result3}")
        print(f"Set Tracking: {result4}")
        print(f"Matrix Tracking: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Visible_Buildings()

