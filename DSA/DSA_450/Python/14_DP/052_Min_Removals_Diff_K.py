"""
Problem: Minimum Removals to Make Max-Min <= K
URL: https://www.geeksforgeeks.org/minimum-removals-array-make-max-min-k/

Problem Statement:
Given an array and a number k, find the minimum number of elements to remove so that the difference between maximum and minimum remaining elements is at most k.

Sample Input/Output:
Input: arr = [1,3,4,9,10,11,12,17,20], k = 4
Output: 5
"""


class Solution:
    def Min_Remove_Memo(self, arr: list[int], i: int, j: int, k: int, dp: list[list[int]]) -> int:
        """
        Memoization approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        if i >= j:
            return 0
        if arr[j] - arr[i] <= k:
            return 0
        if dp[i][j] != -1:
            return dp[i][j]
        
        dp[i][j] = 1 + min(self.Min_Remove_Memo(arr, i + 1, j, k, dp),
                          self.Min_Remove_Memo(arr, i, j - 1, k, dp))
        return dp[i][j]
    
    def Min_Remove_Binary_Search(self, arr: list[int], k: int) -> int:
        """
        Binary search approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        n = len(arr)
        arr.sort()
        
        min_removals = n - 1
        
        for i in range(n):
            left = i
            right = n - 1
            max_idx = i
            
            while left <= right:
                mid = left + (right - left) // 2
                if arr[mid] - arr[i] <= k:
                    max_idx = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            min_removals = min(min_removals, n - (max_idx - i + 1))
        
        return min_removals


def Test_MinRemovalsDiffK():
    solution = Solution()
    
    arr = [1, 3, 4, 9, 10, 11, 12, 17, 20]
    k = 4
    dp = [[-1] * len(arr) for _ in range(len(arr))]
    result1 = solution.Min_Remove_Memo(arr, 0, len(arr) - 1, k, dp)
    assert result1 == 5
    
    arr2 = [1, 3, 4, 9, 10, 11, 12, 17, 20]
    result2 = solution.Min_Remove_Binary_Search(arr2, k)
    assert result2 == 5


if __name__ == "__main__":
    Test_MinRemovalsDiffK()
