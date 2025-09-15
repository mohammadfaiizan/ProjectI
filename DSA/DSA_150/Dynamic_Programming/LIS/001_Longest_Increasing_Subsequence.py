"""
Problem: Longest Increasing Subsequence
URL: https://leetcode.com/problems/longest-increasing-subsequence/

Problem Statement:
Given an integer array nums, return the length of the longest strictly increasing subsequence.

Sample Input/Output:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,18], therefore the length is 4.

Input: nums = [0,1,0,3,2,3]
Output: 4
Explanation: The longest increasing subsequence is [0,1,2,3], therefore the length is 4.
"""

from typing import List
import bisect

class Solution:
    def Length_Of_LIS_Recursive(self, nums: List[int]) -> int:
        """
        Recursive Brute Force - Try all subsequences
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        def LIS_Helper(index: int, prev_val: int) -> int:
            if index >= len(nums):
                return 0
            
            exclude = LIS_Helper(index + 1, prev_val)
            include = 0
            
            if nums[index] > prev_val:
                include = 1 + LIS_Helper(index + 1, nums[index])
            
            return max(include, exclude)
        
        return LIS_Helper(0, float('-inf'))
    
    def Length_Of_LIS_Memoized(self, nums: List[int]) -> int:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        memo = {}
        
        def LIS_Helper(index: int, prev_index: int) -> int:
            if index >= len(nums):
                return 0
            
            if (index, prev_index) in memo:
                return memo[(index, prev_index)]
            
            exclude = LIS_Helper(index + 1, prev_index)
            include = 0
            
            if prev_index == -1 or nums[index] > nums[prev_index]:
                include = 1 + LIS_Helper(index + 1, index)
            
            memo[(index, prev_index)] = max(include, exclude)
            return memo[(index, prev_index)]
        
        return LIS_Helper(0, -1)
    
    def Length_Of_LIS_Tabulation(self, nums: List[int]) -> int:
        """
        Tabulation DP - Bottom-up O(n²) approach
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def Length_Of_LIS_Binary_Search_Optimal(self, nums: List[int]) -> int:
        """
        Binary Search Optimal - O(n log n) approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        tails = []
        
        for num in nums:
            pos = bisect.bisect_left(tails, num)
            
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        
        return len(tails)
    
    def Length_Of_LIS_With_Sequence(self, nums: List[int]) -> tuple:
        """
        With Sequence Tracking - Return length and actual LIS
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not nums:
            return 0, []
        
        n = len(nums)
        dp = [1] * n
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        
        max_length = max(dp)
        max_index = dp.index(max_length)
        
        lis = []
        current = max_index
        
        while current != -1:
            lis.append(nums[current])
            current = parent[current]
        
        return max_length, lis[::-1]
    
    def Length_Of_LIS_Count_All(self, nums: List[int]) -> tuple:
        """
        Count All LIS - Return length and count of all LIS
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not nums:
            return 0, 0
        
        n = len(nums)
        dp = [1] * n
        count = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]
        
        max_length = max(dp)
        total_count = sum(count[i] for i in range(n) if dp[i] == max_length)
        
        return max_length, total_count

def Test_Length_Of_LIS():
    solution = Solution()
    
    test_cases = [
        ([10,9,2,5,3,7,101,18], 4),
        ([0,1,0,3,2,3], 4),
        ([7,7,7,7,7,7,7], 1),
        ([1,3,6,7,9,4,10,5,6], 6),
        ([10,22,9,33,21,50,41,60], 5)
    ]
    
    methods = [
        ("Recursive", solution.Length_Of_LIS_Recursive),
        ("Memoized", solution.Length_Of_LIS_Memoized),
        ("Tabulation", solution.Length_Of_LIS_Tabulation),
        ("Binary Search Optimal", solution.Length_Of_LIS_Binary_Search_Optimal)
    ]
    
    for nums, expected in test_cases:
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive" and len(nums) > 15:
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(nums.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        length, sequence = solution.Length_Of_LIS_With_Sequence(nums.copy())
        print(f"With Sequence: Length={length}, LIS={sequence}")
        
        length, count = solution.Length_Of_LIS_Count_All(nums.copy())
        print(f"Count All: Length={length}, Count={count}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Length_Of_LIS()
