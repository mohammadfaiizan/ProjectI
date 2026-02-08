"""
Problem: Smallest Sum Contiguous Subarray
URL: https://www.geeksforgeeks.org/smallest-sum-contiguous-subarray/

Problem Statement:
Given an array containing n integers. The problem is to find the sum of the elements of the contiguous subarray having the smallest sum.

Sample Input/Output:
Input: [3,-4,2,-3,-1,7,-5]
Output: -6
"""


class Solution:
    def Min_Subarray_Kadane(self, arr: list[int]) -> int:
        """
        Modified Kadane's Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        min_sum = arr[0]
        current_sum = arr[0]
        
        for i in range(1, n):
            current_sum = min(arr[i], current_sum + arr[i])
            min_sum = min(min_sum, current_sum)
        
        return min_sum


def Test_SmallestSumSubarray():
    solution = Solution()
    arr = [3, -4, 2, -3, -1, 7, -5]
    assert solution.Min_Subarray_Kadane(arr) == -6


if __name__ == "__main__":
    Test_SmallestSumSubarray()
