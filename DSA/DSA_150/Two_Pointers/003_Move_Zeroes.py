"""
Problem: Move Zeroes
URL: https://leetcode.com/problems/move-zeroes/description/

Problem Statement:
Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.
Note that you must do this in-place without making a copy of the array.

Sample Input/Output:
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]

Input: nums = [0]
Output: [0]

Input: nums = [0,0,1]
Output: [1,0,0]
"""

from typing import List

class Solution:
    def Move_Zeroes_Extra_Space(self, nums: List[int]) -> None:
        """
        Extra Space - Create new array with non-zeros first
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        non_zeros = []
        zero_count = 0
        
        for num in nums:
            if num != 0:
                non_zeros.append(num)
            else:
                zero_count += 1
        
        for i in range(len(non_zeros)):
            nums[i] = non_zeros[i]
        
        for i in range(len(non_zeros), len(nums)):
            nums[i] = 0
    
    def Move_Zeroes_Two_Pointers_Optimal(self, nums: List[int]) -> None:
        """
        Two Pointers Optimal - One pointer tracks non-zero position
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        write_index = 0
        
        for read_index in range(len(nums)):
            if nums[read_index] != 0:
                nums[write_index] = nums[read_index]
                write_index += 1
        
        for i in range(write_index, len(nums)):
            nums[i] = 0
    
    def Move_Zeroes_Swap_Approach(self, nums: List[int]) -> None:
        """
        Swap Approach - Swap non-zeros with zeros from left
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left = 0
        
        for right in range(len(nums)):
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
    
    def Move_Zeroes_Optimized_Swap(self, nums: List[int]) -> None:
        """
        Optimized Swap - Only swap when necessary
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left = 0
        
        for right in range(len(nums)):
            if nums[right] != 0:
                if left != right:
                    nums[left], nums[right] = nums[right], nums[left]
                left += 1
    
    def Move_Zeroes_Bubble_Style(self, nums: List[int]) -> None:
        """
        Bubble Style - Bubble zeros to the right
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(nums)
        
        for i in range(n):
            for j in range(n - 1):
                if nums[j] == 0 and nums[j + 1] != 0:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
    
    def Move_Zeroes_Partition_Style(self, nums: List[int]) -> None:
        """
        Partition Style - Partition around zero value
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            if nums[left] == 0:
                while right > left and nums[right] == 0:
                    right -= 1
                if right > left:
                    nums[left], nums[right] = nums[right], nums[left]
                    right -= 1
            left += 1

def Test_Move_Zeroes():
    solution = Solution()
    
    test_cases = [
        ([0,1,0,3,12], [1,3,12,0,0]),
        ([0], [0]),
        ([0,0,1], [1,0,0]),
        ([1,2,3], [1,2,3]),
        ([0,0,0], [0,0,0]),
        ([1,0,2,0,3,0,4], [1,2,3,4,0,0,0])
    ]
    
    methods = [
        ("Extra Space", solution.Move_Zeroes_Extra_Space),
        ("Two Pointers Optimal", solution.Move_Zeroes_Two_Pointers_Optimal),
        ("Swap Approach", solution.Move_Zeroes_Swap_Approach),
        ("Optimized Swap", solution.Move_Zeroes_Optimized_Swap),
        ("Bubble Style", solution.Move_Zeroes_Bubble_Style),
        ("Partition Style", solution.Move_Zeroes_Partition_Style)
    ]
    
    for nums, expected in test_cases:
        print(f"Original: {nums}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            test_nums = nums.copy()
            method(test_nums)
            print(f"{method_name}: {test_nums}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Move_Zeroes()
