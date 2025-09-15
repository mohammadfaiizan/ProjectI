"""
Problem: Maximum Length of Repeated Subarray
URL: https://leetcode.com/problems/maximum-length-of-repeated-subarray/

Problem Statement:
Given two integer arrays nums1 and nums2, return the maximum length of a subarray that appears in both arrays.

Sample Input/Output:
Input: nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
Output: 3
Explanation: The repeated subarray with maximum length is [3,2,1].

Input: nums1 = [0,0,0,0,0], nums2 = [0,0,0,0,0]
Output: 5

Input: nums1 = [1,2,3], nums2 = [4,5,6]
Output: 0
"""

from typing import List

class Solution:
    def Find_Length_Brute_Force(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Brute Force - Check all possible subarrays
        Time Complexity: O(m * n * min(m, n))
        Space Complexity: O(1)
        """
        max_length = 0
        m, n = len(nums1), len(nums2)
        
        for i in range(m):
            for j in range(n):
                length = 0
                
                while (i + length < m and j + length < n and 
                       nums1[i + length] == nums2[j + length]):
                    length += 1
                
                max_length = max(max_length, length)
        
        return max_length
    
    def Find_Length_DP_2D_Optimal(self, nums1: List[int], nums2: List[int]) -> int:
        """
        2D DP Optimal - Classic DP approach for longest common subarray
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_length = max(max_length, dp[i][j])
                else:
                    dp[i][j] = 0
        
        return max_length
    
    def Find_Length_Space_Optimized(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Space Optimized DP - Use 1D array
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        m, n = len(nums1), len(nums2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        max_length = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i-1] == nums2[j-1]:
                    curr[j] = prev[j-1] + 1
                    max_length = max(max_length, curr[j])
                else:
                    curr[j] = 0
            
            prev, curr = curr, prev
        
        return max_length
    
    def Find_Length_Rolling_Hash(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Rolling Hash - Use binary search with rolling hash
        Time Complexity: O((m + n) * log(min(m, n)))
        Space Complexity: O(m + n)
        """
        def Get_Rolling_Hashes(arr: List[int], length: int) -> set:
            if length == 0:
                return {0}
            
            base = 113
            mod = 10**9 + 7
            
            hashes = set()
            current_hash = 0
            power = 1
            
            for i in range(length):
                current_hash = (current_hash * base + arr[i]) % mod
                if i < length - 1:
                    power = (power * base) % mod
            
            hashes.add(current_hash)
            
            for i in range(length, len(arr)):
                current_hash = (current_hash - arr[i - length] * power) % mod
                current_hash = (current_hash * base + arr[i]) % mod
                hashes.add(current_hash)
            
            return hashes
        
        def Has_Common_Subarray(length: int) -> bool:
            hashes1 = Get_Rolling_Hashes(nums1, length)
            hashes2 = Get_Rolling_Hashes(nums2, length)
            return bool(hashes1 & hashes2)
        
        left, right = 0, min(len(nums1), len(nums2))
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            if Has_Common_Subarray(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def Find_Length_Suffix_Array(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Suffix Array - Combine arrays and find LCP
        Time Complexity: O((m + n) * log(m + n))
        Space Complexity: O(m + n)
        """
        separator = max(max(nums1, default=0), max(nums2, default=0)) + 1
        combined = nums1 + [separator] + nums2
        n = len(combined)
        
        suffixes = [(combined[i:], i) for i in range(n)]
        suffixes.sort()
        
        max_length = 0
        
        for i in range(n - 1):
            suffix1, pos1 = suffixes[i]
            suffix2, pos2 = suffixes[i + 1]
            
            is_from_different_arrays = ((pos1 < len(nums1)) != (pos2 < len(nums1)))
            
            if is_from_different_arrays:
                lcp_length = 0
                min_len = min(len(suffix1), len(suffix2))
                
                for j in range(min_len):
                    if suffix1[j] == suffix2[j] and suffix1[j] != separator:
                        lcp_length += 1
                    else:
                        break
                
                max_length = max(max_length, lcp_length)
        
        return max_length
    
    def Find_Length_With_Subarray(self, nums1: List[int], nums2: List[int]) -> tuple:
        """
        With Subarray Tracking - Return length and actual subarray
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(nums1), len(nums2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        end_pos_i, end_pos_j = 0, 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_pos_i, end_pos_j = i, j
                else:
                    dp[i][j] = 0
        
        if max_length == 0:
            return 0, []
        
        subarray = nums1[end_pos_i - max_length:end_pos_i]
        return max_length, subarray

def Test_Find_Length():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,2,1], [3,2,1,4,7], 3),
        ([0,0,0,0,0], [0,0,0,0,0], 5),
        ([1,2,3], [4,5,6], 0),
        ([1,0,1,0,1], [1,0,1,1,1], 3),
        ([0,1,1,1,1], [1,0,1,0,1], 2)
    ]
    
    methods = [
        ("Brute Force", solution.Find_Length_Brute_Force),
        ("2D DP Optimal", solution.Find_Length_DP_2D_Optimal),
        ("Space Optimized", solution.Find_Length_Space_Optimized),
        ("Rolling Hash", solution.Find_Length_Rolling_Hash),
        ("Suffix Array", solution.Find_Length_Suffix_Array)
    ]
    
    for nums1, nums2, expected in test_cases:
        print(f"Nums1: {nums1}")
        print(f"Nums2: {nums2}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                result = method(nums1.copy(), nums2.copy())
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        length, subarray = solution.Find_Length_With_Subarray(nums1.copy(), nums2.copy())
        print(f"With Subarray: Length={length}, Subarray={subarray}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Length()
