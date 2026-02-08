"""
Problem: Array Subset of Another Array
URL: https://practice.geeksforgeeks.org/problems/array-subset-of-another-array2317/1

Problem Statement:
Given two arrays a1[] and a2[], determine if a2[] is a subset of a1[].
Both arrays can have duplicates.

Sample Input/Output:
Input: a1 = [11, 1, 13, 21, 3, 7], a2 = [11, 3, 7, 1]
Output: Yes

Input: a1 = [10, 5, 2, 23, 19], a2 = [19, 5, 3]
Output: No
"""


class Solution:
    def Is_Subset_HashSet_Optimal(self, a1, a2):
        """
        HashSet Approach - Check all elements of a2 exist in a1
        Time Complexity: O(n + m)
        Space Complexity: O(n)
        """
        s = set(a1)
        for x in a2:
            if x not in s:
                return "No"
        return "Yes"

    def Is_Subset_HashMap(self, a1, a2):
        """
        HashMap Approach - Handle duplicate frequencies
        Time Complexity: O(n + m)
        Space Complexity: O(n)
        """
        freq = {}
        for x in a1:
            freq[x] = freq.get(x, 0) + 1
        for x in a2:
            if freq.get(x, 0) <= 0:
                return "No"
            freq[x] -= 1
        return "Yes"

    def Is_Subset_Sorting(self, a1, a2):
        """
        Sorting + Two Pointers - Sort both and compare
        Time Complexity: O(n log n + m log m)
        Space Complexity: O(1)
        """
        a1_sorted = sorted(a1)
        a2_sorted = sorted(a2)
        i = 0
        j = 0
        while i < len(a1_sorted) and j < len(a2_sorted):
            if a1_sorted[i] == a2_sorted[j]:
                i += 1
                j += 1
            elif a1_sorted[i] < a2_sorted[j]:
                i += 1
            else:
                return "No"
        return "Yes" if j == len(a2_sorted) else "No"


def Test_Array_Subset():
    solution = Solution()

    test_cases = [
        ([11, 1, 13, 21, 3, 7], [11, 3, 7, 1], "Yes"),
        ([1, 2, 3, 4, 5, 6], [1, 2, 4], "Yes"),
        ([10, 5, 2, 23, 19], [19, 5, 3], "No")
    ]

    for a1, a2, expected in test_cases:
        print(f"a1: {a1}, a2: {a2}, Expected: {expected}")
        result_hashset = solution.Is_Subset_HashSet_Optimal(a1, a2)
        result_hashmap = solution.Is_Subset_HashMap(a1, a2)
        result_sorting = solution.Is_Subset_Sorting(a1, a2)
        print(f"HashSet: {result_hashset}")
        print(f"HashMap: {result_hashmap}")
        print(f"Sorting: {result_sorting}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Array_Subset()
