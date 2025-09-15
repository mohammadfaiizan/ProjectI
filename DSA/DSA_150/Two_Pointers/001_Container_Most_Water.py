"""
Problem: Container With Most Water
URL: https://leetcode.com/problems/container-with-most-water/description/

Problem Statement:
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
Find two lines that together with the x-axis form a container, such that the container contains the most water.
Return the maximum amount of water a container can store.

Sample Input/Output:
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. 
In this case, the max area of water (blue section) the container can contain is 49.

Input: height = [1,1]
Output: 1
Explanation: Container formed by heights 1 and 1 has area 1.
"""

from typing import List

class Solution:
    def Max_Area_Brute_Force(self, height: List[int]) -> int:
        """
        Brute Force - Check all possible pairs
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        max_area = 0
        n = len(height)
        
        for i in range(n):
            for j in range(i + 1, n):
                width = j - i
                min_height = min(height[i], height[j])
                area = width * min_height
                max_area = max(max_area, area)
        
        return max_area
    
    def Max_Area_Two_Pointers_Optimal(self, height: List[int]) -> int:
        """
        Two Pointers Optimal - Start from both ends and move inward
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(height) - 1
        max_area = 0
        
        while left < right:
            width = right - left
            min_height = min(height[left], height[right])
            area = width * min_height
            max_area = max(max_area, area)
            
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
    
    def Max_Area_Optimized_Movement(self, height: List[int]) -> int:
        """
        Optimized Movement - Move pointer with smaller height
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(height) - 1
        max_area = 0
        
        while left < right:
            if height[left] < height[right]:
                area = height[left] * (right - left)
                max_area = max(max_area, area)
                left += 1
            else:
                area = height[right] * (right - left)
                max_area = max(max_area, area)
                right -= 1
        
        return max_area
    
    def Max_Area_Skip_Smaller(self, height: List[int]) -> int:
        """
        Skip Smaller Heights - Skip consecutive smaller values
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(height) - 1
        max_area = 0
        
        while left < right:
            area = min(height[left], height[right]) * (right - left)
            max_area = max(max_area, area)
            
            if height[left] < height[right]:
                current_height = height[left]
                while left < right and height[left] <= current_height:
                    left += 1
            else:
                current_height = height[right]
                while left < right and height[right] <= current_height:
                    right -= 1
        
        return max_area
    
    def Max_Area_Divide_Conquer(self, height: List[int]) -> int:
        """
        Divide and Conquer - Recursive approach
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        """
        def Max_Area_Helper(left: int, right: int) -> int:
            if left >= right:
                return 0
            
            if right - left == 1:
                return min(height[left], height[right]) * (right - left)
            
            mid = (left + right) // 2
            
            left_max = Max_Area_Helper(left, mid)
            right_max = Max_Area_Helper(mid + 1, right)
            
            cross_max = 0
            for i in range(left, mid + 1):
                for j in range(mid + 1, right + 1):
                    area = min(height[i], height[j]) * (j - i)
                    cross_max = max(cross_max, area)
            
            return max(left_max, right_max, cross_max)
        
        return Max_Area_Helper(0, len(height) - 1)

def Test_Max_Area():
    solution = Solution()
    
    test_cases = [
        ([1,8,6,2,5,4,8,3,7], 49),
        ([1,1], 1),
        ([4,3,2,1,4], 16),
        ([1,2,1], 2),
        ([1,2,4,3], 4)
    ]
    
    for height, expected in test_cases:
        result1 = solution.Max_Area_Brute_Force(height.copy())
        result2 = solution.Max_Area_Two_Pointers_Optimal(height.copy())
        result3 = solution.Max_Area_Optimized_Movement(height.copy())
        result4 = solution.Max_Area_Skip_Smaller(height.copy())
        result5 = solution.Max_Area_Divide_Conquer(height.copy())
        
        print(f"Height: {height}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Two Pointers Optimal: {result2}")
        print(f"Optimized Movement: {result3}")
        print(f"Skip Smaller: {result4}")
        print(f"Divide Conquer: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Area()
