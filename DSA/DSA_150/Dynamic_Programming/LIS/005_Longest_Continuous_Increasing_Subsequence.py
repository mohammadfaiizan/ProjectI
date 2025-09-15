"""
Problem: Longest Continuous Increasing Subsequence
URL: https://leetcode.com/problems/longest-continuous-increasing-subsequence/

Problem Statement:
Given an unsorted array of integers nums, return the length of the longest continuous increasing subsequence (i.e. subarray). The subsequence must be strictly increasing.
A continuous increasing subsequence is defined by two indices l and r (l < r) such that it is [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] and for each l <= i < r, nums[i] < nums[i + 1].

Sample Input/Output:
Input: nums = [1,3,5,4,7]
Output: 3
Explanation: The longest continuous increasing subsequence is [1,3,5] with length 3.
Even though [1,3,5,7] is an increasing subsequence, it is not continuous as elements 5 and 7 are separated by element 4.

Input: nums = [2,2,2,2,2]
Output: 1
Explanation: The longest continuous increasing subsequence is [2] with length 1. Note that it must be strictly increasing.
"""

from typing import List

class Solution:
    def Find_Length_Of_LCIS_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force - Check all possible starting positions
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        max_length = 1
        n = len(nums)
        
        for i in range(n):
            current_length = 1
            
            for j in range(i + 1, n):
                if nums[j] > nums[j - 1]:
                    current_length += 1
                else:
                    break
            
            max_length = max(max_length, current_length)
        
        return max_length
    
    def Find_Length_Of_LCIS_One_Pass_Optimal(self, nums: List[int]) -> int:
        """
        One Pass Optimal - Track current and maximum length
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        max_length = 1
        current_length = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                current_length += 1
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        
        return max(max_length, current_length)
    
    def Find_Length_Of_LCIS_Two_Pointers(self, nums: List[int]) -> int:
        """
        Two Pointers - Use sliding window approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not nums:
            return 0
        
        max_length = 1
        left = 0
        
        for right in range(1, len(nums)):
            if nums[right] <= nums[right - 1]:
                left = right
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def Find_Length_Of_LCIS_DP(self, nums: List[int]) -> int:
        """
        Dynamic Programming - DP approach for practice
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n
        
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                dp[i] = dp[i - 1] + 1
        
        return max(dp)
    
    def Find_Length_Of_LCIS_Recursive(self, nums: List[int]) -> int:
        """
        Recursive - Recursive approach with memoization
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        memo = {}
        
        def LCIS_From_Index(index: int) -> int:
            if index >= len(nums):
                return 0
            
            if index in memo:
                return memo[index]
            
            current_length = 1
            
            if index + 1 < len(nums) and nums[index + 1] > nums[index]:
                current_length += LCIS_From_Index(index + 1)
            
            memo[index] = current_length
            return current_length
        
        max_length = 0
        
        for i in range(len(nums)):
            max_length = max(max_length, LCIS_From_Index(i))
        
        return max_length
    
    def Find_Length_Of_LCIS_With_Subarray(self, nums: List[int]) -> tuple:
        """
        With Subarray - Return length and actual LCIS
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0, []
        
        max_length = 1
        current_length = 1
        max_start = 0
        current_start = 0
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    max_start = current_start
                current_length = 1
                current_start = i
        
        if current_length > max_length:
            max_length = current_length
            max_start = current_start
        
        lcis = nums[max_start:max_start + max_length]
        return max_length, lcis
    
    def Find_Length_Of_LCIS_All_Subarrays(self, nums: List[int]) -> tuple:
        """
        All Subarrays - Find all LCIS with maximum length
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0, []
        
        all_lcis = []
        current_start = 0
        current_length = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                current_length += 1
            else:
                all_lcis.append((current_length, nums[current_start:current_start + current_length]))
                current_length = 1
                current_start = i
        
        all_lcis.append((current_length, nums[current_start:current_start + current_length]))
        
        max_length = max(length for length, _ in all_lcis)
        max_lcis = [subarray for length, subarray in all_lcis if length == max_length]
        
        return max_length, max_lcis

def Test_Find_Length_Of_LCIS():
    solution = Solution()
    
    test_cases = [
        ([1,3,5,4,7], 3),
        ([2,2,2,2,2], 1),
        ([1,3,5,7], 4),
        ([2,1,3,4,5], 4),
        ([1,2,1,3,4,5,6], 4),
        ([5,4,3,2,1], 1),
        ([1], 1),
        ([], 0)
    ]
    
    methods = [
        ("Brute Force", solution.Find_Length_Of_LCIS_Brute_Force),
        ("One Pass Optimal", solution.Find_Length_Of_LCIS_One_Pass_Optimal),
        ("Two Pointers", solution.Find_Length_Of_LCIS_Two_Pointers),
        ("DP", solution.Find_Length_Of_LCIS_DP),
        ("Recursive", solution.Find_Length_Of_LCIS_Recursive)
    ]
    
    for nums, expected in test_cases:
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if nums:
            length, lcis = solution.Find_Length_Of_LCIS_With_Subarray(nums.copy())
            print(f"With Subarray: Length={length}, LCIS={lcis}")
            
            length, all_lcis = solution.Find_Length_Of_LCIS_All_Subarrays(nums.copy())
            print(f"All Subarrays: Length={length}, Count={len(all_lcis)}")
            for lcis in all_lcis:
                print(f"  {lcis}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Length_Of_LCIS()
