"""
Problem: Median of Two Sorted Arrays of Equal Size
URL: https://www.geeksforgeeks.org/median-of-two-sorted-arrays/

Problem Statement:
Given two sorted arrays ar1[] and ar2[] of the same size n, find the median of the
merged array (without actually merging). Median is average of elements at n-1 and n.

Sample Input/Output:
Input: ar1 = [1, 12, 15, 26, 38], ar2 = [2, 13, 17, 30, 45]
Output: 16
Explanation: Merged = [1,2,12,13,15,17,26,30,38,45], median = (15+17)/2 = 16.

Input: ar1 = [1, 2, 3, 6], ar2 = [4, 6, 8, 10]
Output: 5
Explanation: Merged = [1,2,3,4,6,6,8,10], median = (4+6)/2 = 5.
"""


class Solution:
    def Median_Binary_Search_Optimal(self, ar1, ar2):
        """
        Divide and Conquer - Compare medians and recurse on halves
        Time Complexity: O(log n)
        Space Complexity: O(log n) recursion stack
        """
        return self.Find_Median_Recursive(ar1, ar2, 0, 0, len(ar1))

    def Median_Merge_Count(self, ar1, ar2):
        """
        Merge Count - Count while merging until reaching median position
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(ar1)
        i = j = 0
        m1 = m2 = -1
        for count in range(n + 1):
            m2 = m1
            if i == n:
                m1 = ar2[j]
                j += 1
            elif j == n:
                m1 = ar1[i]
                i += 1
            elif ar1[i] <= ar2[j]:
                m1 = ar1[i]
                i += 1
            else:
                m1 = ar2[j]
                j += 1
        return (m1 + m2) // 2

    def Median_Full_Merge(self, ar1, ar2):
        """
        Full Merge - Merge both arrays and find median
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(ar1)
        merged = []
        i = j = 0
        while i < n and j < n:
            if ar1[i] <= ar2[j]:
                merged.append(ar1[i])
                i += 1
            else:
                merged.append(ar2[j])
                j += 1
        while i < n:
            merged.append(ar1[i])
            i += 1
        while j < n:
            merged.append(ar2[j])
            j += 1
        return (merged[n - 1] + merged[n]) // 2

    def Median_Single(self, arr, start, n):
        if n % 2 == 0:
            return (arr[start + n // 2] + arr[start + n // 2 - 1]) // 2
        return arr[start + n // 2]

    def Find_Median_Recursive(self, ar1, ar2, s1, s2, n):
        if n <= 0:
            return -1
        if n == 1:
            return (ar1[s1] + ar2[s2]) // 2
        if n == 2:
            return (max(ar1[s1], ar2[s2]) + min(ar1[s1 + 1], ar2[s2 + 1])) // 2

        m1 = self.Median_Single(ar1, s1, n)
        m2 = self.Median_Single(ar2, s2, n)
        if m1 == m2:
            return m1

        half = (n // 2 - 1) if n % 2 == 0 else n // 2
        if m1 < m2:
            return self.Find_Median_Recursive(ar1, ar2, s1 + half, s2, n - half)
        return self.Find_Median_Recursive(ar1, ar2, s1, s2 + half, n - half)


def Test_Median_Equal_Size():
    solution = Solution()

    class TestCase:
        def __init__(self, ar1, ar2, expected):
            self.ar1 = ar1
            self.ar2 = ar2
            self.expected = expected

    test_cases = [
        TestCase([1, 12, 15, 26, 38], [2, 13, 17, 30, 45], 16),
        TestCase([1, 2, 3, 6], [4, 6, 8, 10], 5)
    ]

    for tc in test_cases:
        print(f"ar1: {tc.ar1}, ar2: {tc.ar2}, Expected: {tc.expected}")

        print("Binary Search:", solution.Median_Binary_Search_Optimal(tc.ar1, tc.ar2))
        print("Merge Count:", solution.Median_Merge_Count(tc.ar1, tc.ar2))
        print("Full Merge:", solution.Median_Full_Merge(tc.ar1, tc.ar2))

        print("-" * 50)


if __name__ == "__main__":
    Test_Median_Equal_Size()
