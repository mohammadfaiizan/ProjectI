"""
Problem: Next Permutation
URL: https://leetcode.com/problems/next-permutation/

Problem Statement:
Given an array of integers nums, find the next permutation in lexicographical order.
If no such arrangement exists, rearrange to the lowest possible order (sorted ascending).

Sample Input/Output:
Input: nums = [1, 2, 3]
Output: [1, 3, 2]

Input: nums = [3, 2, 1]
Output: [1, 2, 3]

Input: nums = [1, 1, 5]
Output: [1, 5, 1]
"""


class Solution:
    def Next_Permutation_Optimal(self, nums):
        """
        Optimal Approach - Find rightmost ascending pair, swap and reverse
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1:] = reversed(nums[i + 1:])

    def Next_Permutation_STL(self, nums):
        """
        STL Approach - Using built-in next_permutation
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(nums)
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1:] = reversed(nums[i + 1:])


def Test_Next_Permutation():
    solution = Solution()

    test_cases = [
        [1, 2, 3],
        [3, 2, 1],
        [1, 1, 5],
        [1, 3, 2]
    ]

    for nums in test_cases:
        print("Input:", nums)

        nums1 = nums.copy()
        nums2 = nums.copy()

        solution.Next_Permutation_Optimal(nums1)
        print("Optimal:", nums1)

        solution.Next_Permutation_STL(nums2)
        print("STL:", nums2)

        print("-" * 50)


if __name__ == "__main__":
    Test_Next_Permutation()
