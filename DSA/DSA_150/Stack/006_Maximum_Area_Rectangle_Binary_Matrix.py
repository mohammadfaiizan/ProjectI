"""
Problem: Maximum Area of Rectangle in a Binary Matrix
URL: https://leetcode.com/problems/maximal-rectangle/description/

Problem Statement:
Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

Sample Input/Output:
Input: matrix = [["1","0","1","0","0"],
                 ["1","0","1","1","1"],
                 ["1","1","1","1","1"],
                 ["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.

Input: matrix = [["0"]]
Output: 0
Explanation: No rectangle of 1's possible
"""

from typing import List

class Solution:
    def Maximal_Rectangle_Brute_Force(self, matrix: List[List[str]]) -> int:
        """
        Brute Force Approach - Check all possible rectangles
        Time Complexity: O(m² × n²)
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        max_area = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    for k in range(i, m):
                        for l in range(j, n):
                            if self.Is_Valid_Rectangle(matrix, i, j, k, l):
                                area = (k - i + 1) * (l - j + 1)
                                max_area = max(max_area, area)
        
        return max_area
    
    def Is_Valid_Rectangle(self, matrix: List[List[str]], r1: int, c1: int, r2: int, c2: int) -> bool:
        for i in range(r1, r2 + 1):
            for j in range(c1, c2 + 1):
                if matrix[i][j] == '0':
                    return False
        return True
    
    def Maximal_Rectangle_Dynamic_Programming(self, matrix: List[List[str]]) -> int:
        """
        Dynamic Programming Approach - Build height array
        Time Complexity: O(m × n²)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        max_area = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            area = self.Largest_Rectangle_Area_Brute_Force(heights)
            max_area = max(max_area, area)
        
        return max_area
    
    def Largest_Rectangle_Area_Brute_Force(self, heights: List[int]) -> int:
        max_area = 0
        n = len(heights)
        
        for i in range(n):
            min_height = heights[i]
            for j in range(i, n):
                min_height = min(min_height, heights[j])
                area = min_height * (j - i + 1)
                max_area = max(max_area, area)
        
        return max_area
    
    def Maximal_Rectangle_Stack_Optimal(self, matrix: List[List[str]]) -> int:
        """
        Stack Approach - Optimal solution using histogram algorithm
        Time Complexity: O(m × n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        max_area = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            area = self.Largest_Rectangle_In_Histogram(heights)
            max_area = max(max_area, area)
        
        return max_area
    
    def Largest_Rectangle_In_Histogram(self, heights: List[int]) -> int:
        stack = []
        max_area = 0
        index = 0
        
        while index < len(heights):
            if not stack or heights[index] >= heights[stack[-1]]:
                stack.append(index)
                index += 1
            else:
                top = stack.pop()
                area = (heights[top] * 
                       ((index - stack[-1] - 1) if stack else index))
                max_area = max(max_area, area)
        
        while stack:
            top = stack.pop()
            area = (heights[top] * 
                   ((index - stack[-1] - 1) if stack else index))
            max_area = max(max_area, area)
        
        return max_area
    
    def Maximal_Rectangle_Monotonic_Stack(self, matrix: List[List[str]]) -> int:
        """
        Monotonic Stack Approach - Clean implementation
        Time Complexity: O(m × n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        max_area = 0
        
        for i in range(m):
            for j in range(n):
                heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
            
            max_area = max(max_area, self.Max_Rectangle_Histogram(heights))
        
        return max_area
    
    def Max_Rectangle_Histogram(self, heights: List[int]) -> int:
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
    
    def Maximal_Rectangle_Stack_Enhanced(self, matrix: List[List[str]]) -> int:
        """
        Enhanced Stack Approach - With optimizations
        Time Complexity: O(m × n)
        Space Complexity: O(n)
        """
        if not matrix:
            return 0
        
        n = len(matrix[0])
        heights = [0] * (n + 1)
        max_area = 0
        
        for row in matrix:
            for i in range(n):
                heights[i] = heights[i] + 1 if row[i] == '1' else 0
            
            stack = []
            for i in range(n + 1):
                while stack and heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, h * w)
                stack.append(i)
        
        return max_area

def Test_Maximal_Rectangle():
    solution = Solution()
    
    test_cases = [
        ([["1","0","1","0","0"],
          ["1","0","1","1","1"],
          ["1","1","1","1","1"],
          ["1","0","0","1","0"]], 6),
        ([["0"]], 0),
        ([["1"]], 1),
        ([["1","1","1","1"],
          ["1","1","1","1"],
          ["1","1","1","1"]], 12)
    ]
    
    for matrix, expected in test_cases:
        result1 = solution.Maximal_Rectangle_Brute_Force([row.copy() for row in matrix])
        result2 = solution.Maximal_Rectangle_Dynamic_Programming([row.copy() for row in matrix])
        result3 = solution.Maximal_Rectangle_Stack_Optimal([row.copy() for row in matrix])
        result4 = solution.Maximal_Rectangle_Monotonic_Stack([row.copy() for row in matrix])
        result5 = solution.Maximal_Rectangle_Stack_Enhanced([row.copy() for row in matrix])
        
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Dynamic Programming: {result2}")
        print(f"Stack Optimal: {result3}")
        print(f"Monotonic Stack: {result4}")
        print(f"Stack Enhanced: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Maximal_Rectangle()
