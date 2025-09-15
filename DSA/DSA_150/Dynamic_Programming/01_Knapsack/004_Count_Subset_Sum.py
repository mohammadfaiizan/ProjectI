"""
Problem: Count of Subset Sum with a given Sum
URL: https://leetcode.com/discuss/post/2034613/count-subset-sum-with-a-given-sum-dp-sol-l1pc/

Problem Statement:
Given an array arr[] of length N and an integer X, the task is to find the number of subsets with a sum equal to X.

Sample Input/Output:
Input: arr = [1, 2, 3, 3], sum = 6
Output: 3
Explanation: The subsets are {1, 2, 3}, {1, 2, 3}, and {3, 3}.

Input: arr = [1, 1, 1, 1], sum = 1
Output: 4
Explanation: Each element forms a subset with sum 1.
"""

from typing import List

class Solution:
    def Perfect_Sum_Recursive(self, arr: List[int], n: int, target_sum: int) -> int:
        """
        Recursive Brute Force - Count all subset combinations
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if target_sum == 0:
            return 1
        
        if n == 0:
            return 0
        
        if arr[n-1] > target_sum:
            return self.Perfect_Sum_Recursive(arr, n-1, target_sum)
        
        include = self.Perfect_Sum_Recursive(arr, n-1, target_sum - arr[n-1])
        exclude = self.Perfect_Sum_Recursive(arr, n-1, target_sum)
        
        return include + exclude
    
    def Perfect_Sum_Memoized(self, arr: List[int], n: int, target_sum: int) -> int:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        memo = {}
        
        def Count_Helper(index: int, remaining: int) -> int:
            if remaining == 0:
                return 1
            
            if index == 0:
                return 0
            
            if (index, remaining) in memo:
                return memo[(index, remaining)]
            
            if arr[index-1] > remaining:
                result = Count_Helper(index-1, remaining)
            else:
                include = Count_Helper(index-1, remaining - arr[index-1])
                exclude = Count_Helper(index-1, remaining)
                result = include + exclude
            
            memo[(index, remaining)] = result
            return result
        
        return Count_Helper(n, target_sum)
    
    def Perfect_Sum_Tabulation_Optimal(self, arr: List[int], n: int, target_sum: int) -> int:
        """
        Tabulation DP Optimal - Bottom-up approach
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        dp = [[0 for _ in range(target_sum + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = 1
        
        for i in range(1, n + 1):
            for j in range(target_sum + 1):
                if arr[i-1] <= j:
                    dp[i][j] = dp[i-1][j] + dp[i-1][j - arr[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[n][target_sum]
    
    def Perfect_Sum_Space_Optimized(self, arr: List[int], n: int, target_sum: int) -> int:
        """
        Space Optimized DP - Use 1D array
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        """
        dp = [0] * (target_sum + 1)
        dp[0] = 1
        
        for i in range(n):
            for j in range(target_sum, arr[i] - 1, -1):
                dp[j] += dp[j - arr[i]]
        
        return dp[target_sum]
    
    def Perfect_Sum_Handle_Zeros(self, arr: List[int], n: int, target_sum: int) -> int:
        """
        Handle Zeros - Special handling for zero elements
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        """
        zero_count = arr.count(0)
        non_zero_arr = [x for x in arr if x != 0]
        
        if target_sum == 0:
            return 2 ** zero_count
        
        dp = [0] * (target_sum + 1)
        dp[0] = 1
        
        for num in non_zero_arr:
            for j in range(target_sum, num - 1, -1):
                dp[j] += dp[j - num]
        
        return dp[target_sum] * (2 ** zero_count)
    
    def Perfect_Sum_With_Subsets(self, arr: List[int], n: int, target_sum: int) -> tuple:
        """
        With Subsets Tracking - Return count and actual subsets
        Time Complexity: O(n * sum * result_count)
        Space Complexity: O(n * sum * result_count)
        """
        all_subsets = []
        
        def Find_All_Subsets(index: int, current_subset: List[int], remaining: int) -> None:
            if remaining == 0:
                all_subsets.append(current_subset[:])
                return
            
            if index >= len(arr) or remaining < 0:
                return
            
            current_subset.append(arr[index])
            Find_All_Subsets(index + 1, current_subset, remaining - arr[index])
            current_subset.pop()
            
            Find_All_Subsets(index + 1, current_subset, remaining)
        
        Find_All_Subsets(0, [], target_sum)
        return len(all_subsets), all_subsets

def Test_Perfect_Sum():
    solution = Solution()
    
    test_cases = [
        ([1, 2, 3, 3], 4, 6, 3),
        ([1, 1, 1, 1], 4, 1, 4),
        ([2, 3, 5, 6, 8, 10], 6, 10, 3),
        ([1, 0, 1], 3, 1, 4),
        ([5, 2, 3, 10, 6, 8], 6, 10, 3)
    ]
    
    methods = [
        ("Recursive", solution.Perfect_Sum_Recursive),
        ("Memoized", solution.Perfect_Sum_Memoized),
        ("Tabulation Optimal", solution.Perfect_Sum_Tabulation_Optimal),
        ("Space Optimized", solution.Perfect_Sum_Space_Optimized),
        ("Handle Zeros", solution.Perfect_Sum_Handle_Zeros)
    ]
    
    for arr, n, target_sum, expected in test_cases:
        print(f"Array: {arr}, Target Sum: {target_sum}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                if method_name == "Recursive" and n > 15:
                    print(f"{method_name}: Skipped (too slow)")
                    continue
                
                result = method(arr.copy(), n, target_sum)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if n <= 6:
            count, subsets = solution.Perfect_Sum_With_Subsets(arr.copy(), n, target_sum)
            print(f"With Subsets: Count={count}")
            for subset in subsets:
                print(f"  {subset}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Perfect_Sum()
