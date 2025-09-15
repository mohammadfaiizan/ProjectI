"""
Problem: Minimum Subset Sum Difference
URL: https://leetcode.com/discuss/post/2034794/minimum-subset-sum-difference-explained-aj2u0/

Problem Statement:
Given a set of integers, the task is to divide it into two sets S1 and S2 such that the absolute difference between their sums is minimum.

Sample Input/Output:
Input: arr = [1, 6, 11, 5]
Output: 1
Explanation: Subset1 = {1, 5, 6}, sum of Subset1 = 12 
Subset2 = {11}, sum of Subset2 = 11        
|12-11| = 1

Input: arr = [1, 4]
Output: 3
Explanation: Subset1 = {1}, sum of Subset1 = 1
Subset2 = {4}, sum of Subset2 = 4
|1-4| = 3
"""

from typing import List

class Solution:
    def Find_Min_Recursive(self, arr: List[int], n: int) -> int:
        """
        Recursive Brute Force - Try all subset combinations
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        total_sum = sum(arr)
        
        def Min_Diff_Helper(index: int, sum1: int, sum2: int) -> int:
            if index == n:
                return abs(sum1 - sum2)
            
            include_in_s1 = Min_Diff_Helper(index + 1, sum1 + arr[index], sum2)
            include_in_s2 = Min_Diff_Helper(index + 1, sum1, sum2 + arr[index])
            
            return min(include_in_s1, include_in_s2)
        
        return Min_Diff_Helper(0, 0, 0)
    
    def Find_Min_Memoized(self, arr: List[int], n: int) -> int:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        total_sum = sum(arr)
        memo = {}
        
        def Min_Diff_Helper(index: int, current_sum: int) -> int:
            if index == n:
                return abs(current_sum - (total_sum - current_sum))
            
            if (index, current_sum) in memo:
                return memo[(index, current_sum)]
            
            include = Min_Diff_Helper(index + 1, current_sum + arr[index])
            exclude = Min_Diff_Helper(index + 1, current_sum)
            
            memo[(index, current_sum)] = min(include, exclude)
            return memo[(index, current_sum)]
        
        return Min_Diff_Helper(0, 0)
    
    def Find_Min_Tabulation_Optimal(self, arr: List[int], n: int) -> int:
        """
        Tabulation DP Optimal - Bottom-up subset sum approach
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        total_sum = sum(arr)
        
        dp = [[False for _ in range(total_sum + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, total_sum + 1):
                if arr[i-1] <= j:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - arr[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        min_diff = float('inf')
        
        for j in range(total_sum // 2 + 1):
            if dp[n][j]:
                min_diff = min(min_diff, total_sum - 2 * j)
        
        return min_diff
    
    def Find_Min_Space_Optimized(self, arr: List[int], n: int) -> int:
        """
        Space Optimized DP - Use 1D array
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        """
        total_sum = sum(arr)
        
        dp = [False] * (total_sum + 1)
        dp[0] = True
        
        for num in arr:
            for j in range(total_sum, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        
        min_diff = float('inf')
        
        for j in range(total_sum // 2 + 1):
            if dp[j]:
                min_diff = min(min_diff, total_sum - 2 * j)
        
        return min_diff
    
    def Find_Min_Bitset(self, arr: List[int], n: int) -> int:
        """
        Bitset Approach - Use bitwise operations
        Time Complexity: O(n * sum / 32)
        Space Complexity: O(sum / 32)
        """
        total_sum = sum(arr)
        bits = 1
        
        for num in arr:
            bits |= bits << num
        
        min_diff = total_sum
        
        for j in range(total_sum // 2 + 1):
            if bits & (1 << j):
                min_diff = min(min_diff, total_sum - 2 * j)
        
        return min_diff
    
    def Find_Min_With_Subsets(self, arr: List[int], n: int) -> tuple:
        """
        With Subsets Tracking - Return difference and actual subsets
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        total_sum = sum(arr)
        
        dp = [[False for _ in range(total_sum + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, total_sum + 1):
                if arr[i-1] <= j:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - arr[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        min_diff = float('inf')
        best_sum = 0
        
        for j in range(total_sum // 2 + 1):
            if dp[n][j]:
                diff = total_sum - 2 * j
                if diff < min_diff:
                    min_diff = diff
                    best_sum = j
        
        subset1 = []
        i, j = n, best_sum
        
        while i > 0 and j > 0:
            if dp[i][j] and not dp[i-1][j]:
                subset1.append(arr[i-1])
                j -= arr[i-1]
            i -= 1
        
        subset2 = [x for x in arr if x not in subset1]
        
        return min_diff, subset1, subset2

def Test_Find_Min():
    solution = Solution()
    
    test_cases = [
        ([1, 6, 11, 5], 4, 1),
        ([1, 4], 2, 3),
        ([1, 6, 5, 11], 4, 1),
        ([1, 3, 5, 6], 4, 1),
        ([3, 1, 4, 2, 2, 1], 6, 1)
    ]
    
    methods = [
        ("Recursive", solution.Find_Min_Recursive),
        ("Memoized", solution.Find_Min_Memoized),
        ("Tabulation Optimal", solution.Find_Min_Tabulation_Optimal),
        ("Space Optimized", solution.Find_Min_Space_Optimized),
        ("Bitset", solution.Find_Min_Bitset)
    ]
    
    for arr, n, expected in test_cases:
        print(f"Array: {arr}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive" and n > 15:
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(arr.copy(), n)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        min_diff, subset1, subset2 = solution.Find_Min_With_Subsets(arr.copy(), n)
        print(f"With Subsets: Min Diff={min_diff}")
        print(f"  Subset1: {subset1}, Sum: {sum(subset1)}")
        print(f"  Subset2: {subset2}, Sum: {sum(subset2)}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Min()
