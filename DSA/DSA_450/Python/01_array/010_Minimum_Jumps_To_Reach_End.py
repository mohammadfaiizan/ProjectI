"""
Problem: Minimum Number of Jumps to Reach End
URL: https://practice.geeksforgeeks.org/problems/minimum-number-of-jumps-1587115620/1

Problem Statement:
Given an array of N integers where each element represents the max length of the jump
that can be made forward from that element. Find the minimum number of jumps to reach
the end of the array. Return -1 if end is not reachable.

Sample Input/Output:
Input: arr = [1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9]
Output: 3
Explanation: Jump 1->3->9->last.

Input: arr = [1, 4, 3, 2, 6, 7]
Output: 2
Explanation: Jump 1->4->last.
"""


class Solution:
    def Min_Jumps_Greedy_Optimal(self, arr):
        """
        Greedy Approach - Track max reachable position
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        if n <= 1:
            return 0
        if arr[0] == 0:
            return -1
        max_reach = arr[0]
        steps = arr[0]
        jumps = 1
        for i in range(1, n):
            if i == n - 1:
                return jumps
            max_reach = max(max_reach, i + arr[i])
            steps -= 1
            if steps == 0:
                jumps += 1
                if i >= max_reach:
                    return -1
                steps = max_reach - i
        return -1

    def Min_Jumps_DP(self, arr):
        """
        Dynamic Programming - Build dp array for min jumps to each index
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        n = len(arr)
        if n <= 1:
            return 0
        if arr[0] == 0:
            return -1
        dp = [float('inf')] * n
        dp[0] = 0
        for i in range(1, n):
            for j in range(i):
                if dp[j] != float('inf') and j + arr[j] >= i:
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[n - 1] if dp[n - 1] != float('inf') else -1


def Test_Minimum_Jumps():
    solution = Solution()

    test_cases = [
        ([1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9], 3),
        ([1, 4, 3, 2, 6, 7], 2),
        ([0, 1, 2], -1),
        ([1], 0),
        ([2, 3, 1, 1, 4], 2)
    ]

    for arr, expected in test_cases:
        print(f"Array: {arr}, Expected: {expected}")
        result_greedy = solution.Min_Jumps_Greedy_Optimal(arr)
        result_dp = solution.Min_Jumps_DP(arr)
        print(f"Greedy: {result_greedy}")
        print(f"DP: {result_dp}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Minimum_Jumps()
