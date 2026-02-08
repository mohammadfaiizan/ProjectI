"""
Problem: Next Permutation (Next Greater Number with Same Set of Digits)
URL: https://practice.geeksforgeeks.org/problems/next-permutation5226/1

Problem Statement:
Given a number represented as an array of digits, find the next greater number
using the same set of digits. If no greater permutation exists, return the
smallest permutation.

Sample Input/Output:
Input: [1, 2, 3]
Output: [1, 3, 2]

Input: [3, 2, 1]
Output: [1, 2, 3]

Input: [1, 1, 5]
Output: [1, 5, 1]
"""


class Solution:
    def Next_Permutation_Optimal(self, nums):
        """
        1. Find rightmost element smaller than its next
        2. Swap with smallest element larger than it on right
        3. Reverse the suffix
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
        Using itertools (simulating STL next_permutation)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        import itertools
        n = len(nums)
        for perm in itertools.permutations(nums):
            perm_list = list(perm)
            if perm_list > nums:
                nums[:] = perm_list
                return
        nums.sort()

    def Next_Greater_Number_String(self, num):
        """
        String version of next permutation
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(num)
        num_list = list(num)
        i = n - 2
        while i >= 0 and num_list[i] >= num_list[i + 1]:
            i -= 1

        if i < 0:
            return ''.join(reversed(num_list))

        j = n - 1
        while num_list[j] <= num_list[i]:
            j -= 1
        num_list[i], num_list[j] = num_list[j], num_list[i]
        num_list[i + 1:] = reversed(num_list[i + 1:])
        return ''.join(num_list)


def Test_Next_Permutation():
    sol = Solution()
    tests = [
        [1, 2, 3],
        [3, 2, 1],
        [1, 1, 5],
        [1, 3, 5, 4, 2],
        [5, 4, 3, 2, 1]
    ]

    for nums in tests:
        print(f"Input: {nums}")

        v1 = nums[:]
        sol.Next_Permutation_Optimal(v1)
        print(f"Optimal: {v1}")

        v2 = nums[:]
        sol.Next_Permutation_STL(v2)
        print(f"STL: {v2}")

        print('-' * 50)

    str_tests = ["1234", "4321", "534976"]
    for s in str_tests:
        print(f"String Input: {s}")
        print(f"Next Greater: {sol.Next_Greater_Number_String(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Next_Permutation()
