"""
Problem: Stickler Thief / Max Sum No Two Adjacent
URL: https://practice.geeksforgeeks.org/problems/stickler-theif-1587115621/1

Problem Statement:
Find the maximum sum of a subsequence such that no two elements are adjacent.

Sample Input/Output:
Input: arr[] = {5, 5, 10, 100, 10, 5}
Output: 110

Input: arr[] = {1, 2, 3}
Output: 4
"""


class Solution:
    def Max_Sum_DP_Array(self, arr, n):
        """
        Dynamic programming with array to store maximum sum up to each index
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n == 0:
            return 0
        if n == 1:
            return arr[0]
        
        dp = [0] * n
        dp[0] = arr[0]
        dp[1] = max(arr[0], arr[1])
        
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + arr[i])
        
        return dp[n - 1]

    def Max_Sum_DP_Two_Variables(self, arr, n):
        """
        Dynamic programming using only two variables to track previous two states
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 0
        if n == 1:
            return arr[0]
        
        prev2 = arr[0]
        prev1 = max(arr[0], arr[1])
        
        for i in range(2, n):
            current = max(prev1, prev2 + arr[i])
            prev2 = prev1
            prev1 = current
        
        return prev1

    def Max_Sum_Recursive_Memo(self, arr, n):
        """
        Recursive solution with memoization
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        memo = [-1] * n
        
        def solve(idx):
            if idx >= n:
                return 0
            if memo[idx] != -1:
                return memo[idx]
            
            take = arr[idx] + solve(idx + 2)
            skip = solve(idx + 1)
            
            memo[idx] = max(take, skip)
            return memo[idx]
        
        return solve(0)


def Test_Max_Sum_No_Adjacent():
    sol = Solution()
    tests = [
        [5, 5, 10, 100, 10, 5],
        [1, 2, 3],
        [3, 2, 5, 10, 7],
        [1],
        [2, 1, 4, 9]
    ]

    for arr in tests:
        n = len(arr)
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print()
        
        res1 = sol.Max_Sum_DP_Array(arr[:], n)
        res2 = sol.Max_Sum_DP_Two_Variables(arr[:], n)
        res3 = sol.Max_Sum_Recursive_Memo(arr[:], n)
        
        print(f"DP Array: {res1}")
        print(f"DP Two Variables: {res2}")
        print(f"Recursive + Memo: {res3}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Max_Sum_No_Adjacent()
