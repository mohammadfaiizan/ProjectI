"""
Problem: Largest Rectangle in Histogram
URL: https://leetcode.com/problems/largest-rectangle-in-histogram/description/

Problem Statement:
Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, 
return the area of the largest rectangle in the histogram.

Sample Input/Output:
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The largest rectangle is shown in the above histogram with area = 10 units.

Input: heights = [2,4]
Output: 4
Explanation: The largest rectangle has area = 4 units.
"""

from typing import List

class Solution:
    def Largest_Rectangle_Area_Brute_Force(self, heights: List[int]) -> int:
        """
        Brute Force Approach - Check all possible rectangles
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        max_area = 0
        n = len(heights)
        
        for i in range(n):
            min_height = heights[i]
            for j in range(i, n):
                min_height = min(min_height, heights[j])
                area = min_height * (j - i + 1)
                max_area = max(max_area, area)
        
        return max_area
    
    def Largest_Rectangle_Area_Nested_Loop(self, heights: List[int]) -> int:
        """
        Nested Loop Approach - For each bar find left and right boundaries
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        max_area = 0
        n = len(heights)
        
        for i in range(n):
            left = i
            right = i
            
            while left > 0 and heights[left - 1] >= heights[i]:
                left -= 1
            
            while right < n - 1 and heights[right + 1] >= heights[i]:
                right += 1
            
            width = right - left + 1
            area = heights[i] * width
            max_area = max(max_area, area)
        
        return max_area
    
    def Largest_Rectangle_Area_Stack_Optimal(self, heights: List[int]) -> int:
        """
        Stack Approach - Optimal solution using stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
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
    
    def Largest_Rectangle_Area_Monotonic_Stack(self, heights: List[int]) -> int:
        """
        Monotonic Stack Approach - Clean implementation
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
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
    
    def Largest_Rectangle_Area_Stack_Enhanced(self, heights: List[int]) -> int:
        """
        Enhanced Stack Approach - With sentinel values
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        heights = [0] + heights + [0]
        stack = []
        max_area = 0
        
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)
        
        return max_area
    
    def Largest_Rectangle_Area_Divide_Conquer(self, heights: List[int]) -> int:
        """
        Divide and Conquer Approach - Recursive solution
        Time Complexity: O(n log n) average, O(n²) worst
        Space Complexity: O(log n)
        """
        def Find_Max_Area(start: int, end: int) -> int:
            if start > end:
                return 0
            
            min_idx = start
            for i in range(start, end + 1):
                if heights[i] < heights[min_idx]:
                    min_idx = i
            
            area = heights[min_idx] * (end - start + 1)
            left_area = Find_Max_Area(start, min_idx - 1)
            right_area = Find_Max_Area(min_idx + 1, end)
            
            return max(area, left_area, right_area)
        
        return Find_Max_Area(0, len(heights) - 1)

def Test_Largest_Rectangle_Area():
    solution = Solution()
    
    test_cases = [
        ([2,1,5,6,2,3], 10),
        ([2,4], 4),
        ([1,1,1,1,1], 5),
        ([6,2,5,4,5,1,6], 12),
        ([0,9], 9)
    ]
    
    for heights, expected in test_cases:
        result1 = solution.Largest_Rectangle_Area_Brute_Force(heights.copy())
        result2 = solution.Largest_Rectangle_Area_Nested_Loop(heights.copy())
        result3 = solution.Largest_Rectangle_Area_Stack_Optimal(heights.copy())
        result4 = solution.Largest_Rectangle_Area_Monotonic_Stack(heights.copy())
        result5 = solution.Largest_Rectangle_Area_Stack_Enhanced(heights.copy())
        result6 = solution.Largest_Rectangle_Area_Divide_Conquer(heights.copy())
        
        print(f"Heights: {heights}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Nested Loop: {result2}")
        print(f"Stack Optimal: {result3}")
        print(f"Monotonic Stack: {result4}")
        print(f"Stack Enhanced: {result5}")
        print(f"Divide Conquer: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Largest_Rectangle_Area()
