"""
Problem: Longest Arithmetic Subsequence
URL: https://leetcode.com/problems/longest-arithmetic-subsequence/

Problem Statement:
Given an array nums of integers, return the length of the longest arithmetic subsequence in nums.
An arithmetic subsequence is a sequence of numbers such that the difference between consecutive elements is constant.

Sample Input/Output:
Input: nums = [3,6,9,12]
Output: 4
Explanation: The whole array is an arithmetic sequence with steps of length = 3.

Input: nums = [9,4,7,2,10]
Output: 3
Explanation: The longest arithmetic subsequence is [4,7,10].

Input: nums = [20,1,15,3,10,5,8]
Output: 4
Explanation: The longest arithmetic subsequence is [20,15,10,5].
"""

from typing import List

class Solution:
    def Longest_Arith_Seq_Length_Brute_Force(self, nums: List[int]) -> int:
        """
        Brute Force - Check all possible pairs and extend
        Time Complexity: O(n³)
        Space Complexity: O(1)
        """
        n = len(nums)
        max_length = 2
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = nums[j] - nums[i]
                length = 2
                last = nums[j]
                
                for k in range(j + 1, n):
                    if nums[k] - last == diff:
                        length += 1
                        last = nums[k]
                
                max_length = max(max_length, length)
        
        return max_length if n >= 2 else n
    
    def Longest_Arith_Seq_Length_DP_Hash_Optimal(self, nums: List[int]) -> int:
        """
        DP with Hash Map Optimal - Store diff->length mapping for each position
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        if len(nums) <= 2:
            return len(nums)
        
        n = len(nums)
        dp = [{}] * n
        max_length = 2
        
        for i in range(n):
            dp[i] = {}
            
            for j in range(i):
                diff = nums[i] - nums[j]
                
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                
                max_length = max(max_length, dp[i][diff])
        
        return max_length
    
    def Longest_Arith_Seq_Length_Array_DP(self, nums: List[int]) -> int:
        """
        Array DP - Use 2D array with index mapping
        Time Complexity: O(n²)
        Space Complexity: O(n * range)
        """
        if len(nums) <= 2:
            return len(nums)
        
        n = len(nums)
        min_val = min(nums)
        max_val = max(nums)
        offset = max_val - min_val
        
        if offset == 0:
            return n
        
        dp = [[0] * (2 * offset + 1) for _ in range(n)]
        max_length = 2
        
        for i in range(1, n):
            for j in range(i):
                diff = nums[i] - nums[j] + offset
                
                if dp[j][diff] == 0:
                    dp[i][diff] = 2
                else:
                    dp[i][diff] = dp[j][diff] + 1
                
                max_length = max(max_length, dp[i][diff])
        
        return max_length
    
    def Longest_Arith_Seq_Length_Optimized_Space(self, nums: List[int]) -> int:
        """
        Optimized Space - Use dict with coordinate compression
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        if len(nums) <= 2:
            return len(nums)
        
        n = len(nums)
        max_length = 2
        
        for i in range(n):
            diff_count = {}
            
            for j in range(i + 1, n):
                diff = nums[j] - nums[i]
                
                if diff in diff_count:
                    diff_count[diff] += 1
                else:
                    diff_count[diff] = 2
                
                max_length = max(max_length, diff_count[diff])
                
                for k in range(j + 1, n):
                    if nums[k] - nums[j] == diff:
                        diff_count[diff] += 1
                        max_length = max(max_length, diff_count[diff])
        
        return max_length
    
    def Longest_Arith_Seq_Length_Backwards_DP(self, nums: List[int]) -> int:
        """
        Backwards DP - Process from right to left
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        if len(nums) <= 2:
            return len(nums)
        
        n = len(nums)
        dp = [{} for _ in range(n)]
        max_length = 2
        
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                diff = nums[j] - nums[i]
                
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                
                max_length = max(max_length, dp[i][diff])
        
        return max_length
    
    def Longest_Arith_Seq_Length_With_Sequence(self, nums: List[int]) -> tuple:
        """
        With Sequence Tracking - Return length and actual sequence
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        if len(nums) <= 2:
            return len(nums), nums[:]
        
        n = len(nums)
        dp = [{}] * n
        parent = [{}] * n
        max_length = 2
        best_end = 0
        best_diff = 0
        
        for i in range(n):
            dp[i] = {}
            parent[i] = {}
            
            for j in range(i):
                diff = nums[i] - nums[j]
                
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                    parent[i][diff] = j
                else:
                    dp[i][diff] = 2
                    parent[i][diff] = j
                
                if dp[i][diff] > max_length:
                    max_length = dp[i][diff]
                    best_end = i
                    best_diff = diff
        
        sequence = []
        current = best_end
        
        while current != -1 and best_diff in parent[current]:
            sequence.append(nums[current])
            if best_diff in parent[current]:
                current = parent[current][best_diff]
            else:
                break
            
            if len(sequence) >= max_length:
                break
        
        return max_length, sequence[::-1]

def Test_Longest_Arith_Seq_Length():
    solution = Solution()
    
    test_cases = [
        ([3,6,9,12], 4),
        ([9,4,7,2,10], 3),
        ([20,1,15,3,10,5,8], 4),
        ([1,2,3,4], 4),
        ([1,1,1,1], 4),
        ([1,7,9,11], 3)
    ]
    
    methods = [
        ("Brute Force", solution.Longest_Arith_Seq_Length_Brute_Force),
        ("DP Hash Optimal", solution.Longest_Arith_Seq_Length_DP_Hash_Optimal),
        ("Array DP", solution.Longest_Arith_Seq_Length_Array_DP),
        ("Optimized Space", solution.Longest_Arith_Seq_Length_Optimized_Space),
        ("Backwards DP", solution.Longest_Arith_Seq_Length_Backwards_DP)
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
        
        length, sequence = solution.Longest_Arith_Seq_Length_With_Sequence(nums.copy())
        print(f"With Sequence: Length={length}, Sequence={sequence}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Longest_Arith_Seq_Length()
