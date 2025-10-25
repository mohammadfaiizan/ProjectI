"""
Problem: Move Zeros To Left
URL: https://www.naukri.com/code360/problems/move-zeros-to-left

Problem Statement:
Given an array of integers, move all zeros to the left while maintaining the relative 
order of non-zero elements.

Sample Input/Output:
Input: nums = [1,0,2,0,3,4]
Output: [0,0,1,2,3,4]

Input: nums = [0,1,0,3,12]
Output: [0,0,1,3,12]

Input: nums = [1,2,3,4,5]
Output: [1,2,3,4,5]
"""

from typing import List

class Solution:
    def Move_Zeros_Left_Extra_Array(self, nums: List[int]) -> List[int]:
        """
        Extra Array Approach - Create new array
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        zeros = []
        non_zeros = []
        
        for num in nums:
            if num == 0:
                zeros.append(num)
            else:
                non_zeros.append(num)
        
        return zeros + non_zeros
    
    def Move_Zeros_Left_Two_Pass(self, nums: List[int]) -> List[int]:
        """
        Two Pass Approach - Count zeros then reconstruct
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        zero_count = nums.count(0)
        
        result = [0] * zero_count
        
        for num in nums:
            if num != 0:
                result.append(num)
        
        return result
    
    def Move_Zeros_Left_In_Place(self, nums: List[int]) -> List[int]:
        """
        In-Place Approach - Shift non-zeros right
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        write_pos = n - 1
        
        for i in range(n - 1, -1, -1):
            if nums[i] != 0:
                nums[write_pos] = nums[i]
                write_pos -= 1
        
        while write_pos >= 0:
            nums[write_pos] = 0
            write_pos -= 1
        
        return nums
    
    def Move_Zeros_Left_Two_Pointer(self, nums: List[int]) -> List[int]:
        """
        Two Pointer Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left = len(nums) - 1
        right = len(nums) - 1
        
        while left >= 0:
            if nums[left] != 0:
                nums[right] = nums[left]
                right -= 1
            left -= 1
        
        while right >= 0:
            nums[right] = 0
            right -= 1
        
        return nums
    
    def Move_Zeros_Left_Stable(self, nums: List[int]) -> List[int]:
        """
        Stable Approach - Maintain order
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        zero_count = 0
        
        for num in nums:
            if num == 0:
                zero_count += 1
        
        idx = len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] != 0:
                nums[idx] = nums[i]
                idx -= 1
        
        for i in range(zero_count):
            nums[i] = 0
        
        return nums

def Test_Move_Zeros_Left():
    solution = Solution()
    
    test_cases = [
        ([1,0,2,0,3,4], [0,0,1,2,3,4]),
        ([0,1,0,3,12], [0,0,1,3,12]),
        ([1,2,3,4,5], [1,2,3,4,5]),
        ([0,0,0,1], [0,0,0,1]),
        ([1,0,0,0], [0,0,0,1]),
        ([5,4,0,3,0,2,1], [0,0,5,4,3,2,1])
    ]
    
    for nums, expected in test_cases:
        result1 = solution.Move_Zeros_Left_Extra_Array(nums.copy())
        result2 = solution.Move_Zeros_Left_Two_Pass(nums.copy())
        result3 = solution.Move_Zeros_Left_In_Place(nums.copy())
        result4 = solution.Move_Zeros_Left_Two_Pointer(nums.copy())
        result5 = solution.Move_Zeros_Left_Stable(nums.copy())
        
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        print(f"Extra Array: {result1}")
        print(f"Two Pass: {result2}")
        print(f"In-Place: {result3}")
        print(f"Two Pointer: {result4}")
        print(f"Stable: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Move_Zeros_Left()

