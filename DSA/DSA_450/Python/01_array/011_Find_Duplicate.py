"""
Problem: Find the Duplicate Number
URL: https://leetcode.com/problems/find-the-duplicate-number/

Problem Statement:
Given an array of integers nums containing n + 1 integers where each integer is in the
range [1, n] inclusive. There is only one repeated number, return this repeated number.

Sample Input/Output:
Input: nums = [1, 3, 4, 2, 2]
Output: 2

Input: nums = [3, 1, 3, 4, 2]
Output: 3
"""


class Solution:
    def Find_Duplicate_Floyd_Cycle_Optimal(self, nums):
        """
        Floyd's Tortoise and Hare - Cycle detection in linked list
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        slow = nums[0]
        fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        fast = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        return slow

    def Find_Duplicate_Negative_Marking(self, nums):
        """
        Negative Marking - Mark visited indices as negative
        Time Complexity: O(n)
        Space Complexity: O(1) - modifies input
        """
        nums_copy = nums[:]
        for i in range(len(nums_copy)):
            idx = abs(nums_copy[i]) - 1
            nums_copy[idx] = -nums_copy[idx]
            if nums_copy[idx] > 0:
                return idx + 1
        return -1

    def Find_Duplicate_Hashing(self, nums):
        """
        Hashing Approach - Use set to detect duplicate
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        seen = set()
        for x in nums:
            if x in seen:
                return x
            seen.add(x)
        return -1


def Test_Find_Duplicate():
    solution = Solution()

    test_cases = [
        ([1, 3, 4, 2, 2], 2),
        ([3, 1, 3, 4, 2], 3),
        ([1, 1], 1),
        ([2, 5, 9, 6, 9, 3, 8, 9, 7, 1], 9)
    ]

    for nums, expected in test_cases:
        print(f"Array: {nums}, Expected: {expected}")
        result_floyd = solution.Find_Duplicate_Floyd_Cycle_Optimal(nums)
        result_negative = solution.Find_Duplicate_Negative_Marking(nums)
        result_hashing = solution.Find_Duplicate_Hashing(nums)
        print(f"Floyd's Cycle: {result_floyd}")
        print(f"Negative Marking: {result_negative}")
        print(f"Hashing: {result_hashing}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Find_Duplicate()
