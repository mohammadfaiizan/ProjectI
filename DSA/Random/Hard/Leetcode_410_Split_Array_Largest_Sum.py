"""
Problem: Split Array Largest Sum
URL: https://leetcode.com/problems/split-array-largest-sum/

Problem Statement:
Given an integer array nums and an integer k, split nums into k non-empty subarrays 
such that the largest sum of any subarray is minimized.

Return the minimized largest sum of the split.

A subarray is a contiguous part of the array.

Sample Input/Output:
Input: nums = [7,2,5,10,8], k = 2
Output: 18
Explanation: Split into [7,2,5] and [10,8]. Largest sum = max(14, 18) = 18.

Input: nums = [1,2,3,4,5], k = 2
Output: 9
Explanation: Split into [1,2,3,4] and [5]. Largest sum = max(10, 5) = 10.
             Better: [1,2,3] and [4,5]. Largest sum = max(6, 9) = 9.

Input: nums = [1,4,4], k = 3
Output: 4
"""

from typing import List

class Solution:
    def Split_Array_DP(self, nums: List[int], k: int) -> int:
        """
        Dynamic Programming Approach
        Time Complexity: O(n^2 * k)
        Space Complexity: O(n * k)
        """
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for p in range(i):
                    subarray_sum = prefix_sum[i] - prefix_sum[p]
                    dp[i][j] = min(dp[i][j], max(dp[p][j - 1], subarray_sum))
        
        return dp[n][k]
    
    def Split_Array_Binary_Search(self, nums: List[int], k: int) -> int:
        """
        Binary Search Approach - Search on answer
        Time Complexity: O(n * log(sum(nums)))
        Space Complexity: O(1)
        """
        def Can_Split(max_sum: int) -> bool:
            subarrays = 1
            current_sum = 0
            
            for num in nums:
                if current_sum + num > max_sum:
                    subarrays += 1
                    current_sum = num
                    if subarrays > k:
                        return False
                else:
                    current_sum += num
            
            return True
        
        left, right = max(nums), sum(nums)
        result = right
        
        while left <= right:
            mid = (left + right) // 2
            
            if Can_Split(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def Split_Array_Binary_Search_Optimal(self, nums: List[int], k: int) -> int:
        """
        Binary Search Optimal - Most efficient
        Time Complexity: O(n * log(sum(nums)))
        Space Complexity: O(1)
        """
        def Min_Subarrays_Needed(max_sum: int) -> int:
            count = 1
            current = 0
            
            for num in nums:
                if current + num > max_sum:
                    count += 1
                    current = num
                else:
                    current += num
            
            return count
        
        left, right = max(nums), sum(nums)
        
        while left < right:
            mid = (left + right) // 2
            
            if Min_Subarrays_Needed(mid) <= k:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def Split_Array_Binary_Search_Greedy(self, nums: List[int], k: int) -> int:
        """
        Binary Search with Greedy Validation
        Time Complexity: O(n * log(sum(nums)))
        Space Complexity: O(1)
        """
        def Is_Valid(target: int) -> bool:
            splits = 1
            total = 0
            
            for num in nums:
                total += num
                if total > target:
                    splits += 1
                    total = num
            
            return splits <= k
        
        lo, hi = max(nums), sum(nums)
        answer = hi
        
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            
            if Is_Valid(mid):
                answer = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        return answer
    
    def Split_Array_Binary_Search_Compact(self, nums: List[int], k: int) -> int:
        """
        Compact Binary Search Implementation
        Time Complexity: O(n * log(sum(nums)))
        Space Complexity: O(1)
        """
        def Check(target: int) -> bool:
            parts, s = 1, 0
            for n in nums:
                s += n
                if s > target:
                    parts += 1
                    s = n
            return parts <= k
        
        left, right = max(nums), sum(nums)
        
        while left < right:
            mid = (left + right) // 2
            if Check(mid):
                right = mid
            else:
                left = mid + 1
        
        return left

def Test_Split_Array():
    solution = Solution()
    
    test_cases = [
        ([7,2,5,10,8], 2, 18),
        ([1,2,3,4,5], 2, 9),
        ([1,4,4], 3, 4),
        ([10,5,13,4,8,4,5,11,14,9,16,10,20,8], 8, 25),
        ([1], 1, 1),
        ([1,1,1,1], 2, 2)
    ]
    
    for nums, k, expected in test_cases:
        result1 = solution.Split_Array_DP(nums.copy(), k)
        result2 = solution.Split_Array_Binary_Search(nums.copy(), k)
        result3 = solution.Split_Array_Binary_Search_Optimal(nums.copy(), k)
        result4 = solution.Split_Array_Binary_Search_Greedy(nums.copy(), k)
        result5 = solution.Split_Array_Binary_Search_Compact(nums.copy(), k)
        
        print(f"Array: {nums}, k: {k}")
        print(f"Expected: {expected}")
        print(f"DP: {result1}")
        print(f"Binary Search: {result2}")
        print(f"Binary Search Optimal: {result3}")
        print(f"Binary Search Greedy: {result4}")
        print(f"Binary Search Compact: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Split_Array()

