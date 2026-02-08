"""
Problem: Search in Rotated Sorted Array
URL: https://leetcode.com/problems/search-in-rotated-sorted-array/

Problem Statement:
There is an integer array nums sorted in ascending order (with distinct values). Prior to being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed).

Sample Input/Output:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
"""


class Solution:
    def Search_Rotated_Find_Pivot_Then_Search(self, nums, target):
        """
        Find pivot point first, then binary search in appropriate half
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        n = len(nums)
        left = 0
        right = n - 1
        
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        
        pivot = left
        left = 0
        right = n - 1
        
        if target >= nums[pivot] and target <= nums[n - 1]:
            left = pivot
        else:
            right = pivot - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1

    def Search_Rotated_Single_Pass(self, nums, target):
        """
        Single pass binary search accounting for rotation
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left = 0
        right = len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[left] <= nums[mid]:
                if target >= nums[left] and target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if target > nums[mid] and target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1


def Test_Search_In_Rotated_Array():
    sol = Solution()
    tests = [
        ([4, 5, 6, 7, 0, 1, 2], 0),
        ([4, 5, 6, 7, 0, 1, 2], 3),
        ([1], 0),
        ([1, 3], 3),
        ([3, 1], 1)
    ]

    for nums, target in tests:
        print("Array:", end=" ")
        for num in nums:
            print(num, end=" ")
        print(f", target = {target}")
        
        res1 = sol.Search_Rotated_Find_Pivot_Then_Search(nums, target)
        print(f"Find Pivot Then Search: {res1}")
        
        res2 = sol.Search_Rotated_Single_Pass(nums, target)
        print(f"Single Pass: {res2}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Search_In_Rotated_Array()
