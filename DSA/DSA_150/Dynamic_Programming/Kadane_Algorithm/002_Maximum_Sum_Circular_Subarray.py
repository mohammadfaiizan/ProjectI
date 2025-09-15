"""
Problem: Maximum Sum Circular Subarray
URL: https://leetcode.com/problems/maximum-sum-circular-subarray/

Problem Statement:
Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.
A circular array means the end of the array connects to the beginning of the array. 
Formally, the next element of nums[i] is nums[(i + 1) % n] and the previous element of nums[i] is nums[(i - 1 + n) % n].
A subarray may only include each element of the fixed buffer nums at most once. 
Formally, for a subarray nums[i], nums[i + 1], ..., nums[j], there are no indices k1, k2 where i <= k1, k2 <= j and k1 % n == k2 % n.

Sample Input/Output:
Input: nums = [1,-2,3,-2]
Output: 3
Explanation: Subarray [3] has maximum sum 3.

Input: nums = [5,-3,5]
Output: 10
Explanation: Subarray [5,5] has maximum sum 5 + 5 = 10.

Input: nums = [-3,-2,-3]
Output: -2
Explanation: Subarray [-2] has maximum sum -2.
"""

from typing import List, Tuple

class Solution:
    def Max_Subarray_Sum_Circular_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force - Check all possible circular subarrays
        Time Complexity: O(n³)
        Space Complexity: O(1)
        """
        n = len(nums)
        max_sum = float('-inf')
        
        for i in range(n):
            for length in range(1, n + 1):
                current_sum = 0
                for k in range(length):
                    current_sum += nums[(i + k) % n]
                max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def Max_Subarray_Sum_Circular_Optimized_Brute_Force(self, nums: List[int]) -> int:
        """
        Optimized Brute Force - Avoid recalculating sums
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(nums)
        max_sum = float('-inf')
        
        for i in range(n):
            current_sum = 0
            for length in range(1, n + 1):
                current_sum += nums[(i + length - 1) % n]
                max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def Max_Subarray_Sum_Circular_Two_Cases_Optimal(self, nums: List[int]) -> int:
        """
        Two Cases Optimal - Non-circular vs circular cases
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        def Kadane_Max(arr: List[int]) -> int:
            max_ending_here = max_so_far = arr[0]
            for i in range(1, len(arr)):
                max_ending_here = max(arr[i], max_ending_here + arr[i])
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        def Kadane_Min(arr: List[int]) -> int:
            min_ending_here = min_so_far = arr[0]
            for i in range(1, len(arr)):
                min_ending_here = min(arr[i], min_ending_here + arr[i])
                min_so_far = min(min_so_far, min_ending_here)
            return min_so_far
        
        case1 = Kadane_Max(nums)
        
        total_sum = sum(nums)
        min_subarray_sum = Kadane_Min(nums)
        case2 = total_sum - min_subarray_sum
        
        if case2 == 0:
            return case1
        
        return max(case1, case2)
    
    def Max_Subarray_Sum_Circular_DP_States(self, nums: List[int]) -> int:
        """
        DP States - Track max/min ending at each position
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(nums)
        
        max_kadane = [0] * n
        min_kadane = [0] * n
        
        max_kadane[0] = min_kadane[0] = nums[0]
        
        for i in range(1, n):
            max_kadane[i] = max(nums[i], max_kadane[i-1] + nums[i])
            min_kadane[i] = min(nums[i], min_kadane[i-1] + nums[i])
        
        max_linear = max(max_kadane)
        min_linear = min(min_kadane)
        
        total_sum = sum(nums)
        max_circular = total_sum - min_linear
        
        if max_circular == 0:
            return max_linear
        
        return max(max_linear, max_circular)
    
    def Max_Subarray_Sum_Circular_Prefix_Suffix(self, nums: List[int]) -> int:
        """
        Prefix Suffix - Use prefix and suffix maximum sums
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(nums)
        
        def Kadane_Max(arr: List[int]) -> int:
            max_ending_here = max_so_far = arr[0]
            for i in range(1, len(arr)):
                max_ending_here = max(arr[i], max_ending_here + arr[i])
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        max_kadane = Kadane_Max(nums)
        
        prefix_max = [0] * n
        suffix_max = [0] * n
        
        prefix_max[0] = nums[0]
        for i in range(1, n):
            prefix_max[i] = prefix_max[i-1] + nums[i]
        
        suffix_max[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix_max[i] = suffix_max[i+1] + nums[i]
        
        for i in range(1, n):
            prefix_max[i] = max(prefix_max[i], prefix_max[i-1])
        
        for i in range(n-2, -1, -1):
            suffix_max[i] = max(suffix_max[i], suffix_max[i+1])
        
        max_circular = float('-inf')
        for i in range(n-1):
            max_circular = max(max_circular, prefix_max[i] + suffix_max[i+1])
        
        return max(max_kadane, max_circular)
    
    def Max_Subarray_Sum_Circular_Deque(self, nums: List[int]) -> int:
        """
        Deque - Using deque for sliding window maximum
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        from collections import deque
        
        n = len(nums)
        
        def Kadane_Max(arr: List[int]) -> int:
            max_ending_here = max_so_far = arr[0]
            for i in range(1, len(arr)):
                max_ending_here = max(arr[i], max_ending_here + arr[i])
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        max_linear = Kadane_Max(nums)
        
        extended = nums + nums
        prefix_sum = [0] * (2 * n + 1)
        for i in range(2 * n):
            prefix_sum[i + 1] = prefix_sum[i] + extended[i]
        
        dq = deque()
        max_circular = float('-inf')
        
        for i in range(2 * n):
            if i >= n:
                while dq and dq[0] <= i - n:
                    dq.popleft()
                
                if dq:
                    max_circular = max(max_circular, prefix_sum[i + 1] - prefix_sum[dq[0]])
            
            while dq and prefix_sum[dq[-1]] >= prefix_sum[i]:
                dq.pop()
            
            dq.append(i)
        
        return max(max_linear, max_circular)
    
    def Max_Subarray_Sum_Circular_With_Indices(self, nums: List[int]) -> Tuple[int, List[int]]:
        """
        With Indices - Return maximum sum and the actual circular subarray
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        def Kadane_Max_With_Indices(arr: List[int]) -> Tuple[int, int, int]:
            max_ending_here = max_so_far = arr[0]
            start = end = 0
            temp_start = 0
            
            for i in range(1, len(arr)):
                if max_ending_here < 0:
                    max_ending_here = arr[i]
                    temp_start = i
                else:
                    max_ending_here += arr[i]
                
                if max_ending_here > max_so_far:
                    max_so_far = max_ending_here
                    start = temp_start
                    end = i
            
            return max_so_far, start, end
        
        def Kadane_Min_With_Indices(arr: List[int]) -> Tuple[int, int, int]:
            min_ending_here = min_so_far = arr[0]
            start = end = 0
            temp_start = 0
            
            for i in range(1, len(arr)):
                if min_ending_here > 0:
                    min_ending_here = arr[i]
                    temp_start = i
                else:
                    min_ending_here += arr[i]
                
                if min_ending_here < min_so_far:
                    min_so_far = min_ending_here
                    start = temp_start
                    end = i
            
            return min_so_far, start, end
        
        max_linear, max_start, max_end = Kadane_Max_With_Indices(nums)
        min_linear, min_start, min_end = Kadane_Min_With_Indices(nums)
        
        total_sum = sum(nums)
        max_circular = total_sum - min_linear
        
        if max_circular == 0 or max_linear >= max_circular:
            return max_linear, nums[max_start:max_end+1]
        else:
            circular_subarray = nums[min_end+1:] + nums[:min_start]
            return max_circular, circular_subarray
    
    def Max_Subarray_Sum_Circular_All_Cases(self, nums: List[int]) -> Tuple[int, List[List[int]]]:
        """
        All Cases - Find all maximum circular subarrays
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        max_sum = self.Max_Subarray_Sum_Circular_Two_Cases_Optimal(nums)
        max_subarrays = []
        n = len(nums)
        
        for i in range(n):
            current_sum = 0
            for length in range(1, n + 1):
                current_sum += nums[(i + length - 1) % n]
                if current_sum == max_sum:
                    subarray = []
                    for k in range(length):
                        subarray.append(nums[(i + k) % n])
                    max_subarrays.append(subarray)
        
        return max_sum, max_subarrays

def Test_Max_Subarray_Sum_Circular():
    solution = Solution()
    
    test_cases = [
        ([1,-2,3,-2], 3),
        ([5,-3,5], 10),
        ([-3,-2,-3], -2),
        ([3,-2,2,-3], 3),
        ([-2,-3,-1], -1),
        ([2,-1,2,2], 6),
        ([1,2,3], 6),
        ([-1,-2,-3], -1)
    ]
    
    methods = [
        ("Optimized Brute Force", solution.Max_Subarray_Sum_Circular_Optimized_Brute_Force),
        ("Two Cases Optimal", solution.Max_Subarray_Sum_Circular_Two_Cases_Optimal),
        ("DP States", solution.Max_Subarray_Sum_Circular_DP_States),
        ("Prefix Suffix", solution.Max_Subarray_Sum_Circular_Prefix_Suffix),
        ("Deque", solution.Max_Subarray_Sum_Circular_Deque)
    ]
    
    for nums, expected in test_cases:
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        if len(nums) <= 6:
            result_bf = solution.Max_Subarray_Sum_Circular_Brute_Force(nums.copy())
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_sum, subarray = solution.Max_Subarray_Sum_Circular_With_Indices(nums.copy())
        print(f"With Indices: Sum={max_sum}, Subarray={subarray}")
        
        if len(nums) <= 6:
            max_sum, all_subarrays = solution.Max_Subarray_Sum_Circular_All_Cases(nums.copy())
            print(f"All Max Subarrays: Sum={max_sum}, Count={len(all_subarrays)}")
            for subarray in all_subarrays[:3]:
                print(f"  {subarray}")
            if len(all_subarrays) > 3:
                print(f"  ... and {len(all_subarrays) - 3} more")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Subarray_Sum_Circular()
