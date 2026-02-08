"""
Problem: Maximum Product Subarray
URL: https://practice.geeksforgeeks.org/problems/maximum-product-subarray3604/1

Problem Statement:
Given an array arr[] that contains positive and negative integers (may contain 0),
find the maximum product subarray.

Sample Input/Output:
Input: arr = [6, -3, -10, 0, 2]
Output: 180
Explanation: Subarray [6, -3, -10] has product 180.

Input: arr = [-1, -3, -10, 0, 60]
Output: 60
Explanation: Subarray [60] has product 60.
"""


class Solution:
    def Max_Product_DP_Optimal(self, arr):
        """
        DP Approach - Track min and max product ending at each index
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        max_val = arr[0]
        min_val = arr[0]
        max_product = arr[0]
        for i in range(1, len(arr)):
            if arr[i] < 0:
                max_val, min_val = min_val, max_val
            max_val = max(arr[i], max_val * arr[i])
            min_val = min(arr[i], min_val * arr[i])
            max_product = max(max_product, max_val)
        return max_product

    def Max_Product_Prefix_Suffix(self, arr):
        """
        Prefix-Suffix Product - Compute products from both ends
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        max_product = float('-inf')
        prefix = 1
        suffix = 1
        for i in range(n):
            prefix *= arr[i]
            suffix *= arr[n - 1 - i]
            max_product = max(max_product, prefix, suffix)
            if prefix == 0:
                prefix = 1
            if suffix == 0:
                suffix = 1
        return max_product

    def Max_Product_Brute_Force(self, arr):
        """
        Brute Force - Check all subarrays
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(arr)
        result = arr[0]
        for i in range(n):
            product = 1
            for j in range(i, n):
                product *= arr[j]
                result = max(result, product)
        return result


def Test_Maximum_Product_Subarray():
    solution = Solution()

    test_cases = [
        ([6, -3, -10, 0, 2], 180),
        ([-1, -3, -10, 0, 60], 60),
        ([2, 3, -2, 4], 6),
        ([-2, 0, -1], 0)
    ]

    for arr, expected in test_cases:
        print(f"Array: {arr}, Expected: {expected}")
        result_dp = solution.Max_Product_DP_Optimal(arr)
        result_prefix = solution.Max_Product_Prefix_Suffix(arr)
        result_brute = solution.Max_Product_Brute_Force(arr)
        print(f"DP: {result_dp}")
        print(f"Prefix-Suffix: {result_prefix}")
        print(f"Brute Force: {result_brute}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Maximum_Product_Subarray()
