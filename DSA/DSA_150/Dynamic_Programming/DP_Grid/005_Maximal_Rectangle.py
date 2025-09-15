"""
Problem: Maximal Rectangle
URL: https://leetcode.com/problems/maximal-rectangle/

Problem Statement:
Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

Sample Input/Output:
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.

Input: matrix = [["0"]]
Output: 0

Input: matrix = [["1"]]
Output: 1
"""

from typing import List, Tuple

class Solution:
    def Maximal_Rectangle_Brute_Force(self, matrix: List[List[str]]) -> int:
        """
        Brute Force - Check all possible rectangles
        Time Complexity: O(m²*n²)
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        max_area = 0
        
        for i1 in range(m):
            for j1 in range(n):
                for i2 in range(i1, m):
                    for j2 in range(j1, n):
                        if self.Is_Valid_Rectangle(matrix, i1, j1, i2, j2):
                            area = (i2 - i1 + 1) * (j2 - j1 + 1)
                            max_area = max(max_area, area)
        
        return max_area
    
    def Is_Valid_Rectangle(self, matrix: List[List[str]], i1: int, j1: int, i2: int, j2: int) -> bool:
        """Helper function to check if rectangle contains only 1s"""
        for i in range(i1, i2 + 1):
            for j in range(j1, j2 + 1):
                if matrix[i][j] == '0':
                    return False
        return True
    
    def Maximal_Rectangle_Histogram_Optimal(self, matrix: List[List[str]]) -> int:
        """
        Histogram Optimal - Convert to largest rectangle in histogram
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        max_area = 0
        
        def Largest_Rectangle_In_Histogram(heights: List[int]) -> int:
            stack = []
            max_area = 0
            
            for i in range(len(heights)):
                while stack and heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, h * w)
                
                stack.append(i)
            
            while stack:
                h = heights[stack.pop()]
                w = len(heights) if not stack else len(heights) - stack[-1] - 1
                max_area = max(max_area, h * w)
            
            return max_area
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            max_area = max(max_area, Largest_Rectangle_In_Histogram(heights))
        
        return max_area
    
    def Maximal_Rectangle_DP_2D(self, matrix: List[List[str]]) -> int:
        """
        DP 2D - Track heights, left and right boundaries
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        left_bounds = [0] * n
        right_bounds = [n] * n
        max_area = 0
        
        for i in range(m):
            current_left = 0
            current_right = n
            
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            for j in range(n):
                if matrix[i][j] == '1':
                    left_bounds[j] = max(left_bounds[j], current_left)
                else:
                    left_bounds[j] = 0
                    current_left = j + 1
            
            for j in range(n - 1, -1, -1):
                if matrix[i][j] == '1':
                    right_bounds[j] = min(right_bounds[j], current_right)
                else:
                    right_bounds[j] = n
                    current_right = j
            
            for j in range(n):
                area = heights[j] * (right_bounds[j] - left_bounds[j])
                max_area = max(max_area, area)
        
        return max_area
    
    def Maximal_Rectangle_Stack_Optimized(self, matrix: List[List[str]]) -> int:
        """
        Stack Optimized - Optimized stack approach
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * (n + 1)
        max_area = 0
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            stack = []
            
            for j in range(n + 1):
                while stack and heights[j] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = j if not stack else j - stack[-1] - 1
                    max_area = max(max_area, h * w)
                
                stack.append(j)
        
        return max_area
    
    def Maximal_Rectangle_Monotonic_Stack(self, matrix: List[List[str]]) -> int:
        """
        Monotonic Stack - Use monotonic stack for each row
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        max_area = 0
        
        def Calculate_Max_Rectangle(heights: List[int]) -> int:
            stack = [-1]
            max_area = 0
            
            for i in range(len(heights)):
                while len(stack) > 1 and heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i - stack[-1] - 1
                    max_area = max(max_area, h * w)
                
                stack.append(i)
            
            while len(stack) > 1:
                h = heights[stack.pop()]
                w = len(heights) - stack[-1] - 1
                max_area = max(max_area, h * w)
            
            return max_area
        
        heights = [0] * n
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            max_area = max(max_area, Calculate_Max_Rectangle(heights))
        
        return max_area
    
    def Maximal_Rectangle_With_Coordinates(self, matrix: List[List[str]]) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
        """
        With Coordinates - Return area and coordinates of maximal rectangle
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0, (-1, -1), (-1, -1)
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        max_area = 0
        best_coords = ((-1, -1), (-1, -1))
        
        def Find_Max_Rectangle_With_Coords(heights: List[int], row: int) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
            stack = []
            max_area = 0
            best_coords = ((-1, -1), (-1, -1))
            
            for i in range(len(heights)):
                while stack and heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    area = h * w
                    
                    if area > max_area:
                        max_area = area
                        left_col = 0 if not stack else stack[-1] + 1
                        right_col = i - 1
                        top_row = row - h + 1
                        bottom_row = row
                        best_coords = ((top_row, left_col), (bottom_row, right_col))
                
                stack.append(i)
            
            while stack:
                h = heights[stack.pop()]
                w = len(heights) if not stack else len(heights) - stack[-1] - 1
                area = h * w
                
                if area > max_area:
                    max_area = area
                    left_col = 0 if not stack else stack[-1] + 1
                    right_col = len(heights) - 1
                    top_row = row - h + 1
                    bottom_row = row
                    best_coords = ((top_row, left_col), (bottom_row, right_col))
            
            return max_area, best_coords[0], best_coords[1]
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            area, top_left, bottom_right = Find_Max_Rectangle_With_Coords(heights, i)
            
            if area > max_area:
                max_area = area
                best_coords = (top_left, bottom_right)
        
        return max_area, best_coords[0], best_coords[1]
    
    def Maximal_Rectangle_Divide_Conquer(self, matrix: List[List[str]]) -> int:
        """
        Divide Conquer - Divide and conquer approach
        Time Complexity: O(m*n*log(n))
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        max_area = 0
        
        def Max_Rectangle_Divide_Conquer(heights: List[int], left: int, right: int) -> int:
            if left > right:
                return 0
            
            min_height = min(heights[left:right+1])
            min_index = heights.index(min_height, left, right+1)
            
            area = min_height * (right - left + 1)
            
            left_area = Max_Rectangle_Divide_Conquer(heights, left, min_index - 1)
            right_area = Max_Rectangle_Divide_Conquer(heights, min_index + 1, right)
            
            return max(area, left_area, right_area)
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            max_area = max(max_area, Max_Rectangle_Divide_Conquer(heights, 0, n - 1))
        
        return max_area
    
    def Maximal_Rectangle_Segment_Tree(self, matrix: List[List[str]]) -> int:
        """
        Segment Tree - Use segment tree for range minimum query
        Time Complexity: O(m*n*log(n))
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        max_area = 0
        
        class SegmentTree:
            def __init__(self, arr: List[int]):
                self.n = len(arr)
                self.tree = [0] * (4 * self.n)
                self.arr = arr
                self.Build(1, 0, self.n - 1)
            
            def Build(self, node: int, start: int, end: int) -> None:
                if start == end:
                    self.tree[node] = start
                else:
                    mid = (start + end) // 2
                    self.Build(2 * node, start, mid)
                    self.Build(2 * node + 1, mid + 1, end)
                    
                    left_idx = self.tree[2 * node]
                    right_idx = self.tree[2 * node + 1]
                    
                    if self.arr[left_idx] <= self.arr[right_idx]:
                        self.tree[node] = left_idx
                    else:
                        self.tree[node] = right_idx
            
            def Query(self, node: int, start: int, end: int, l: int, r: int) -> int:
                if r < start or end < l:
                    return -1
                
                if l <= start and end <= r:
                    return self.tree[node]
                
                mid = (start + end) // 2
                left_idx = self.Query(2 * node, start, mid, l, r)
                right_idx = self.Query(2 * node + 1, mid + 1, end, l, r)
                
                if left_idx == -1:
                    return right_idx
                if right_idx == -1:
                    return left_idx
                
                if self.arr[left_idx] <= self.arr[right_idx]:
                    return left_idx
                else:
                    return right_idx
        
        def Max_Rectangle_Segment_Tree(heights: List[int]) -> int:
            if not heights or all(h == 0 for h in heights):
                return 0
            
            seg_tree = SegmentTree(heights)
            
            def Solve(left: int, right: int) -> int:
                if left > right:
                    return 0
                
                min_idx = seg_tree.Query(1, 0, len(heights) - 1, left, right)
                area = heights[min_idx] * (right - left + 1)
                
                left_area = Solve(left, min_idx - 1)
                right_area = Solve(min_idx + 1, right)
                
                return max(area, left_area, right_area)
            
            return Solve(0, len(heights) - 1)
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            max_area = max(max_area, Max_Rectangle_Segment_Tree(heights))
        
        return max_area

def Test_Maximal_Rectangle():
    solution = Solution()
    
    test_cases = [
        ([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 6),
        ([["0"]], 0),
        ([["1"]], 1),
        ([["1","1"],["1","1"]], 4),
        ([["0","0"],["0","0"]], 0),
        ([["1","1","1","1"],["1","1","1","1"],["1","1","1","1"]], 12),
        ([["1","0","1","1","1"],["0","1","0","1","0"],["1","1","1","1","1"],["1","1","1","1","0"]], 6)
    ]
    
    methods = [
        ("Histogram Optimal", solution.Maximal_Rectangle_Histogram_Optimal),
        ("DP 2D", solution.Maximal_Rectangle_DP_2D),
        ("Stack Optimized", solution.Maximal_Rectangle_Stack_Optimized),
        ("Monotonic Stack", solution.Maximal_Rectangle_Monotonic_Stack),
        ("Divide Conquer", solution.Maximal_Rectangle_Divide_Conquer)
    ]
    
    for matrix, expected in test_cases:
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        
        if len(matrix) <= 3 and len(matrix[0]) <= 3:
            result_bf = solution.Maximal_Rectangle_Brute_Force([row[:] for row in matrix])
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method([row[:] for row in matrix])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        area, top_left, bottom_right = solution.Maximal_Rectangle_With_Coordinates([row[:] for row in matrix])
        print(f"With Coordinates: Area={area}, Top-left={top_left}, Bottom-right={bottom_right}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Maximal_Rectangle()
