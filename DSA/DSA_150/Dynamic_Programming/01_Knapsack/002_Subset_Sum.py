"""
Problem: Subset Sum
URL: https://www.geeksforgeeks.org/problems/subset-sum-problem-1611555638/1

Problem Statement:
Given a set of non-negative integers, and a value sum, determine if there is a subset of the given set with sum equal to given sum.

Sample Input/Output:
Input: N = 6, arr[] = {3, 34, 4, 12, 5, 2}, sum = 9
Output: True
Explanation: Here there exists a subset with sum = 9, 4+3+2 = 9.

Input: N = 6, arr[] = {3, 34, 4, 12, 5, 2}, sum = 30
Output: False
Explanation: There is no subset with sum 30.
"""

from typing import List

class Solution:
    def Is_Subset_Sum_Recursive(self, arr: List[int], n: int, target_sum: int) -> bool:
        """
        Recursive Brute Force - Try all subsets
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if target_sum == 0:
            return True
        
        if n == 0:
            return False
        
        if arr[n-1] > target_sum:
            return self.Is_Subset_Sum_Recursive(arr, n-1, target_sum)
        
        include = self.Is_Subset_Sum_Recursive(arr, n-1, target_sum - arr[n-1])
        exclude = self.Is_Subset_Sum_Recursive(arr, n-1, target_sum)
        
        return include or exclude
    
    def Is_Subset_Sum_Memoized(self, arr: List[int], n: int, target_sum: int) -> bool:
        """
        Memoized DP - Top-down with caching
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        memo = {}
        
        def Subset_Helper(index: int, remaining_sum: int) -> bool:
            if remaining_sum == 0:
                return True
            
            if index == 0:
                return False
            
            if (index, remaining_sum) in memo:
                return memo[(index, remaining_sum)]
            
            if arr[index-1] > remaining_sum:
                result = Subset_Helper(index-1, remaining_sum)
            else:
                include = Subset_Helper(index-1, remaining_sum - arr[index-1])
                exclude = Subset_Helper(index-1, remaining_sum)
                result = include or exclude
            
            memo[(index, remaining_sum)] = result
            return result
        
        return Subset_Helper(n, target_sum)
    
    def Is_Subset_Sum_Tabulation_Optimal(self, arr: List[int], n: int, target_sum: int) -> bool:
        """
        Tabulation DP Optimal - Bottom-up approach
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        dp = [[False for _ in range(target_sum + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target_sum + 1):
                if arr[i-1] <= j:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - arr[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[n][target_sum]
    
    def Is_Subset_Sum_Space_Optimized(self, arr: List[int], n: int, target_sum: int) -> bool:
        """
        Space Optimized DP - Use 1D array
        Time Complexity: O(n * sum)
        Space Complexity: O(sum)
        """
        dp = [False for _ in range(target_sum + 1)]
        dp[0] = True
        
        for i in range(n):
            for j in range(target_sum, arr[i] - 1, -1):
                dp[j] = dp[j] or dp[j - arr[i]]
        
        return dp[target_sum]
    
    def Is_Subset_Sum_Bitset(self, arr: List[int], n: int, target_sum: int) -> bool:
        """
        Bitset Approach - Use bitwise operations
        Time Complexity: O(n * sum / 32)
        Space Complexity: O(sum / 32)
        """
        bits = 1
        
        for num in arr:
            bits |= bits << num
        
        return bool(bits & (1 << target_sum))
    
    def Is_Subset_Sum_With_Subset(self, arr: List[int], n: int, target_sum: int) -> tuple:
        """
        With Subset Tracking - Return the actual subset
        Time Complexity: O(n * sum)
        Space Complexity: O(n * sum)
        """
        dp = [[False for _ in range(target_sum + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = True
        
        for i in range(1, n + 1):
            for j in range(1, target_sum + 1):
                if arr[i-1] <= j:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - arr[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        if not dp[n][target_sum]:
            return False, []
        
        subset = []
        i, j = n, target_sum
        
        while i > 0 and j > 0:
            if dp[i][j] and not dp[i-1][j]:
                subset.append(arr[i-1])
                j -= arr[i-1]
            i -= 1
        
        return True, subset[::-1]

def Test_Is_Subset_Sum():
    solution = Solution()
    
    test_cases = [
        ([3, 34, 4, 12, 5, 2], 6, 9, True),
        ([3, 34, 4, 12, 5, 2], 6, 30, False),
        ([1, 2, 3, 7], 4, 6, True),
        ([1, 2, 7, 1, 5], 5, 10, True),
        ([1, 3, 5], 3, 4, True),
        ([2, 4, 6, 8], 4, 7, False)
    ]
    
    methods = [
        ("Recursive", solution.Is_Subset_Sum_Recursive),
        ("Memoized", solution.Is_Subset_Sum_Memoized),
        ("Tabulation Optimal", solution.Is_Subset_Sum_Tabulation_Optimal),
        ("Space Optimized", solution.Is_Subset_Sum_Space_Optimized),
        ("Bitset", solution.Is_Subset_Sum_Bitset)
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
        
        found, subset = solution.Is_Subset_Sum_With_Subset(arr.copy(), n, target_sum)
        print(f"With Subset: Found={found}, Subset={subset}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Is_Subset_Sum()
