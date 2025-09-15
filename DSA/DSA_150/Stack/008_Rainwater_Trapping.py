"""
Problem: Rainwater Trapping
URL: https://leetcode.com/problems/trapping-rain-water/description/

Problem Statement:
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it can trap after raining.

Sample Input/Output:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
In this case, 6 units of rain water are being trapped.

Input: height = [4,2,0,3,2,5]
Output: 9
Explanation: 9 units of water can be trapped
"""

from typing import List

class Solution:
    def Trap_Water_Brute_Force(self, height: List[int]) -> int:
        """
        Brute Force Approach - For each position find left and right max
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(height)
        total_water = 0
        
        for i in range(1, n - 1):
            left_max = 0
            for j in range(i):
                left_max = max(left_max, height[j])
            
            right_max = 0
            for j in range(i + 1, n):
                right_max = max(right_max, height[j])
            
            water_level = min(left_max, right_max)
            if water_level > height[i]:
                total_water += water_level - height[i]
        
        return total_water
    
    def Trap_Water_Dynamic_Programming(self, height: List[int]) -> int:
        """
        Dynamic Programming Approach - Precompute left and right max arrays
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not height:
            return 0
        
        n = len(height)
        left_max = [0] * n
        right_max = [0] * n
        
        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i - 1], height[i])
        
        right_max[n - 1] = height[n - 1]
        for i in range(n - 2, -1, -1):
            right_max[i] = max(right_max[i + 1], height[i])
        
        total_water = 0
        for i in range(n):
            water_level = min(left_max[i], right_max[i])
            total_water += max(0, water_level - height[i])
        
        return total_water
    
    def Trap_Water_Two_Pointers_Optimal(self, height: List[int]) -> int:
        """
        Two Pointers Approach - Optimal space solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not height:
            return 0
        
        left, right = 0, len(height) - 1
        left_max, right_max = 0, 0
        total_water = 0
        
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    total_water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    total_water += right_max - height[right]
                right -= 1
        
        return total_water
    
    def Trap_Water_Stack_Approach(self, height: List[int]) -> int:
        """
        Stack Approach - Using stack to find water levels
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = []
        total_water = 0
        
        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                bottom = stack.pop()
                
                if not stack:
                    break
                
                distance = i - stack[-1] - 1
                bounded_height = min(height[i], height[stack[-1]]) - height[bottom]
                total_water += distance * bounded_height
            
            stack.append(i)
        
        return total_water
    
    def Trap_Water_Monotonic_Stack(self, height: List[int]) -> int:
        """
        Monotonic Stack Approach - Maintain increasing stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = []
        water = 0
        
        for i, h in enumerate(height):
            while stack and h > height[stack[-1]]:
                bottom = stack.pop()
                if not stack:
                    break
                
                width = i - stack[-1] - 1
                min_height = min(h, height[stack[-1]])
                water += width * (min_height - height[bottom])
            
            stack.append(i)
        
        return water
    
    def Trap_Water_Prefix_Suffix(self, height: List[int]) -> int:
        """
        Prefix Suffix Approach - Alternative DP implementation
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if len(height) <= 2:
            return 0
        
        n = len(height)
        prefix_max = [0] * n
        suffix_max = [0] * n
        
        prefix_max[0] = height[0]
        for i in range(1, n):
            prefix_max[i] = max(prefix_max[i - 1], height[i])
        
        suffix_max[n - 1] = height[n - 1]
        for i in range(n - 2, -1, -1):
            suffix_max[i] = max(suffix_max[i + 1], height[i])
        
        total_water = 0
        for i in range(n):
            water_level = min(prefix_max[i], suffix_max[i])
            total_water += max(0, water_level - height[i])
        
        return total_water

def Test_Trap_Water():
    solution = Solution()
    
    test_cases = [
        ([0,1,0,2,1,0,1,3,2,1,2,1], 6),
        ([4,2,0,3,2,5], 9),
        ([3,0,2,0,4], 7),
        ([1,2,3,4,5], 0),
        ([5,4,3,2,1], 0)
    ]
    
    for height, expected in test_cases:
        result1 = solution.Trap_Water_Brute_Force(height.copy())
        result2 = solution.Trap_Water_Dynamic_Programming(height.copy())
        result3 = solution.Trap_Water_Two_Pointers_Optimal(height.copy())
        result4 = solution.Trap_Water_Stack_Approach(height.copy())
        result5 = solution.Trap_Water_Monotonic_Stack(height.copy())
        result6 = solution.Trap_Water_Prefix_Suffix(height.copy())
        
        print(f"Height: {height}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Dynamic Programming: {result2}")
        print(f"Two Pointers Optimal: {result3}")
        print(f"Stack Approach: {result4}")
        print(f"Monotonic Stack: {result5}")
        print(f"Prefix Suffix: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Trap_Water()
