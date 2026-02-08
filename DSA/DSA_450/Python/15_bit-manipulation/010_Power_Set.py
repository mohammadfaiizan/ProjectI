"""
Problem: Power Set (Generate All Subsets)
URL: https://practice.geeksforgeeks.org/problems/power-set4302/1

Problem Statement:
Given a string/array, generate all possible subsets using bit manipulation.

Sample Input/Output:
Input: "abc"
Output: ["","a","b","ab","c","ac","bc","abc"]

Input: [1,2,3]
Output: All subsets
"""


class Solution:
    def Power_Set_Bitmask(self, s):
        """
        Iterate 0 to 2^n - 1, include element if bit is set
        Time Complexity: O(2^n * n)
        Space Complexity: O(2^n * n)
        """
        n = len(s)
        result = []
        for i in range(1 << n):
            subset = ""
            for j in range(n):
                if i & (1 << j):
                    subset += s[j]
            result.append(subset)
        return result

    def Power_Set_Recursive_Helper(self, s, index, current, result):
        if index == len(s):
            result.append(current)
            return
        self.Power_Set_Recursive_Helper(s, index + 1, current, result)
        self.Power_Set_Recursive_Helper(s, index + 1, current + s[index], result)

    def Power_Set_Recursive(self, s):
        """
        Recursive include/exclude
        Time Complexity: O(2^n * n)
        Space Complexity: O(2^n * n)
        """
        result = []
        self.Power_Set_Recursive_Helper(s, 0, "", result)
        return result

    def Power_Set_Bitmask_Array(self, arr):
        """
        Iterate 0 to 2^n - 1, include element if bit is set
        Time Complexity: O(2^n * n)
        Space Complexity: O(2^n * n)
        """
        n = len(arr)
        result = []
        for i in range(1 << n):
            subset = []
            for j in range(n):
                if i & (1 << j):
                    subset.append(arr[j])
            result.append(subset)
        return result


def Test_Power_Set():
    solution = Solution()

    print("Testing Power_Set_Bitmask:")
    s = "abc"
    result1 = solution.Power_Set_Bitmask(s)
    print("Input: \"abc\"")
    print("Output:", result1)

    print("\nTesting Power_Set_Recursive:")
    result2 = solution.Power_Set_Recursive(s)
    print("Input: \"abc\"")
    print("Output:", result2)

    print("\nTesting Power_Set_Bitmask_Array:")
    arr = [1, 2, 3]
    result3 = solution.Power_Set_Bitmask_Array(arr)
    print("Input: [1,2,3]")
    print("Output:", result3)


if __name__ == "__main__":
    Test_Power_Set()
