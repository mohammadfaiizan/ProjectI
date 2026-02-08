"""
Problem: Find Two Non-Repeating Elements
URL: https://practice.geeksforgeeks.org/problems/finding-the-numbers0702/1

Problem Statement:
Given an array where every element appears twice except two elements, find those two unique elements.

Sample Input/Output:
Input: [2,4,7,9,2,4]
Output: {7,9}

Input: [1,1,2,3,3,4,4,5]
Output: {2,5}
"""

from collections import Counter


class Solution:
    def Non_Repeating_XOR(self, nums):
        """
        XOR approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        xor_all = 0
        for num in nums:
            xor_all ^= num

        rightmost_set_bit = xor_all & (-xor_all)

        group1 = 0
        group2 = 0
        for num in nums:
            if num & rightmost_set_bit:
                group1 ^= num
            else:
                group2 ^= num

        return (min(group1, group2), max(group1, group2))

    def Non_Repeating_Hash(self, nums):
        """
        Frequency map approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        freq = Counter(nums)
        result = []
        for num, count in freq.items():
            if count == 1:
                result.append(num)

        return (min(result[0], result[1]), max(result[0], result[1]))


def Test_Non_Repeating_Elements():
    solution = Solution()

    test1 = [2, 4, 7, 9, 2, 4]
    result1 = solution.Non_Repeating_XOR(test1)
    print("Test 1 XOR:", test1, "->", result1, "(expected: (7, 9))")

    result1_hash = solution.Non_Repeating_Hash(test1)
    print("Test 1 Hash:", test1, "->", result1_hash, "(expected: (7, 9))")

    test2 = [1, 1, 2, 3, 3, 4, 4, 5]
    result2 = solution.Non_Repeating_XOR(test2)
    print("Test 2 XOR:", test2, "->", result2, "(expected: (2, 5))")

    result2_hash = solution.Non_Repeating_Hash(test2)
    print("Test 2 Hash:", test2, "->", result2_hash, "(expected: (2, 5))")


if __name__ == "__main__":
    Test_Non_Repeating_Elements()
