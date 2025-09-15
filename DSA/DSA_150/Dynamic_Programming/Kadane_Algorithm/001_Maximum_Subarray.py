"""
Problem: Maximum Subarray (Kadane's Algorithm)
URL: https://leetcode.com/problems/maximum-subarray/

Problem Statement:
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
A subarray is a contiguous part of an array.

Sample Input/Output:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum = 1.

Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: [5,4,-1,7,8] has the largest sum = 23.
"""

from typing import List, Tuple

class Solution:
    def Max_Subarray_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force - Check all possible subarrays
        Time Complexity: O(n³)
        Space Complexity: O(1)
        """
        n = len(nums)
        max_sum = float('-inf')
        
        for i in range(n):
            for j in range(i, n):
                current_sum = 0
                for k in range(i, j + 1):
                    current_sum += nums[k]
                max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def Max_Subarray_Optimized_Brute_Force(self, nums: List[int]) -> int:
        """
        Optimized Brute Force - Avoid recalculating sums
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(nums)
        max_sum = float('-inf')
        
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def Max_Subarray_Kadane_Classic_Optimal(self, nums: List[int]) -> int:
        """
        Kadane's Algorithm Classic - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_ending_here = max_so_far = nums[0]
        
        for i in range(1, len(nums)):
            max_ending_here = max(nums[i], max_ending_here + nums[i])
            max_so_far = max(max_so_far, max_ending_here)
        
        return max_so_far
    
    def Max_Subarray_DP_Array(self, nums: List[int]) -> int:
        """
        DP Array - Using DP array to store max sum ending at each position
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        max_sum = dp[0]
        
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i-1] + nums[i])
            max_sum = max(max_sum, dp[i])
        
        return max_sum
    
    def Max_Subarray_With_Indices(self, nums: List[int]) -> Tuple[int, int, int]:
        """
        With Indices - Return max sum and start/end indices
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_ending_here = max_so_far = nums[0]
        start = end = 0
        temp_start = 0
        
        for i in range(1, len(nums)):
            if max_ending_here < 0:
                max_ending_here = nums[i]
                temp_start = i
            else:
                max_ending_here += nums[i]
            
            if max_ending_here > max_so_far:
                max_so_far = max_ending_here
                start = temp_start
                end = i
        
        return max_so_far, start, end
    
    def Max_Subarray_Divide_Conquer(self, nums: List[int]) -> int:
        """
        Divide and Conquer - Recursive approach
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        """
        def Max_Crossing_Sum(arr: List[int], left: int, mid: int, right: int) -> int:
            left_sum = float('-inf')
            total = 0
            for i in range(mid, left - 1, -1):
                total += arr[i]
                left_sum = max(left_sum, total)
            
            right_sum = float('-inf')
            total = 0
            for i in range(mid + 1, right + 1):
                total += arr[i]
                right_sum = max(right_sum, total)
            
            return left_sum + right_sum
        
        def Max_Subarray_Rec(arr: List[int], left: int, right: int) -> int:
            if left == right:
                return arr[left]
            
            mid = (left + right) // 2
            
            left_sum = Max_Subarray_Rec(arr, left, mid)
            right_sum = Max_Subarray_Rec(arr, mid + 1, right)
            cross_sum = Max_Crossing_Sum(arr, left, mid, right)
            
            return max(left_sum, right_sum, cross_sum)
        
        return Max_Subarray_Rec(nums, 0, len(nums) - 1)
    
    def Max_Subarray_Prefix_Sum(self, nums: List[int]) -> int:
        """
        Prefix Sum - Using prefix sum approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_sum = float('-inf')
        current_sum = 0
        min_prefix = 0
        
        for num in nums:
            current_sum += num
            max_sum = max(max_sum, current_sum - min_prefix)
            min_prefix = min(min_prefix, current_sum)
        
        return max_sum
    
    def Max_Subarray_Sliding_Window(self, nums: List[int]) -> int:
        """
        Sliding Window - Window-based approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_sum = nums[0]
        window_sum = 0
        
        for num in nums:
            if window_sum < 0:
                window_sum = num
            else:
                window_sum += num
            
            max_sum = max(max_sum, window_sum)
        
        return max_sum
    
    def Max_Subarray_All_Subarrays(self, nums: List[int]) -> Tuple[int, List[List[int]]]:
        """
        All Subarrays - Find all subarrays with maximum sum
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        max_sum = self.Max_Subarray_Kadane_Classic_Optimal(nums)
        max_subarrays = []
        
        n = len(nums)
        
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                if current_sum == max_sum:
                    max_subarrays.append(nums[i:j+1])
        
        return max_sum, max_subarrays
    
    def Max_Subarray_With_Length_Constraint(self, nums: List[int], k: int) -> int:
        """
        With Length Constraint - Maximum subarray with length at most k
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(nums)
        max_sum = float('-inf')
        
        for i in range(n):
            current_sum = 0
            for j in range(i, min(i + k, n)):
                current_sum += nums[j]
                max_sum = max(max_sum, current_sum)
        
        return max_sum

def Test_Max_Subarray():
    solution = Solution()
    
    test_cases = [
        ([-2,1,-3,4,-1,2,1,-5,4], 6),
        ([1], 1),
        ([5,4,-1,7,8], 23),
        ([-1], -1),
        ([-2,-3,-1,-5], -1),
        ([1,2,3,4,5], 15),
        ([-1,-2,3,4,-5,6], 6),
        ([2,-3,4,-1,2,1,-5,3], 6)
    ]
    
    methods = [
        ("Optimized Brute Force", solution.Max_Subarray_Optimized_Brute_Force),
        ("Kadane Classic Optimal", solution.Max_Subarray_Kadane_Classic_Optimal),
        ("DP Array", solution.Max_Subarray_DP_Array),
        ("Divide Conquer", solution.Max_Subarray_Divide_Conquer),
        ("Prefix Sum", solution.Max_Subarray_Prefix_Sum),
        ("Sliding Window", solution.Max_Subarray_Sliding_Window)
    ]
    
    for nums, expected in test_cases:
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        if len(nums) <= 8:
            result_bf = solution.Max_Subarray_Brute_Force(nums.copy())
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_sum, start, end = solution.Max_Subarray_With_Indices(nums.copy())
        print(f"With Indices: Sum={max_sum}, Start={start}, End={end}, Subarray={nums[start:end+1]}")
        
        if len(nums) <= 8:
            max_sum, all_subarrays = solution.Max_Subarray_All_Subarrays(nums.copy())
            print(f"All Max Subarrays: Sum={max_sum}, Count={len(all_subarrays)}")
            for subarray in all_subarrays:
                print(f"  {subarray}")
        
        constrained_result = solution.Max_Subarray_With_Length_Constraint(nums.copy(), 3)
        print(f"Length Constraint (k=3): {constrained_result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Subarray()
