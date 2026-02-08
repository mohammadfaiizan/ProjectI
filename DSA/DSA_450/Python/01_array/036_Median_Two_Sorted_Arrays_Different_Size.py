"""
Problem: Median of Two Sorted Arrays of Different Size
URL: https://www.geeksforgeeks.org/median-of-two-sorted-arrays-of-different-sizes/

Problem Statement:
Given two sorted arrays of different sizes, find the median of the merged array.
If the merged array has even size, median is the average of the two middle elements.

Sample Input/Output:
Input: ar1 = [900], ar2 = [5, 8, 10, 20]
Output: 10
Explanation: Merged = [5, 8, 10, 20, 900], median = 10.

Input: ar1 = [-5, 3, 6, 12, 15], ar2 = [-12, -10, -6, -3, 4, 10]
Output: 3
Explanation: Merged = [-12,-10,-6,-5,-3,3,4,6,10,12,15], median = 3.
"""


class Solution:
    def Median_Binary_Search_Optimal(self, nums1, nums2):
        """
        Binary Search on Smaller Array - Partition both arrays around median
        Time Complexity: O(log(min(n, m)))
        Space Complexity: O(1)
        """
        if len(nums1) > len(nums2):
            return self.Median_Binary_Search_Optimal(nums2, nums1)
        n, m = len(nums1), len(nums2)
        low, high = 0, n
        while low <= high:
            cut1 = (low + high) // 2
            cut2 = (n + m + 1) // 2 - cut1
            left1 = float('-inf') if cut1 == 0 else nums1[cut1 - 1]
            left2 = float('-inf') if cut2 == 0 else nums2[cut2 - 1]
            right1 = float('inf') if cut1 == n else nums1[cut1]
            right2 = float('inf') if cut2 == m else nums2[cut2]
            if left1 <= right2 and left2 <= right1:
                if (n + m) % 2 == 0:
                    return (max(left1, left2) + min(right1, right2)) / 2.0
                return max(left1, left2)
            elif left1 > right2:
                high = cut1 - 1
            else:
                low = cut1 + 1
        return 0.0

    def Median_Merge_Count(self, ar1, ar2):
        """
        Merge and Count - Walk merge to find median position
        Time Complexity: O(n + m)
        Space Complexity: O(1)
        """
        n, m = len(ar1), len(ar2)
        i = j = 0
        m1 = m2 = -1
        target = (n + m) // 2
        for count in range(target + 1):
            m2 = m1
            if i < n and j < m:
                m1 = ar1[i] if ar1[i] <= ar2[j] else ar2[j]
                if ar1[i] <= ar2[j]:
                    i += 1
                else:
                    j += 1
            elif i < n:
                m1 = ar1[i]
                i += 1
            else:
                m1 = ar2[j]
                j += 1
        if (n + m) % 2 == 1:
            return m1
        return (m1 + m2) / 2.0

    def Median_Full_Merge(self, ar1, ar2):
        """
        Full Merge - Merge into new array and find median
        Time Complexity: O(n + m)
        Space Complexity: O(n + m)
        """
        merged = []
        i = j = 0
        while i < len(ar1) and j < len(ar2):
            if ar1[i] <= ar2[j]:
                merged.append(ar1[i])
                i += 1
            else:
                merged.append(ar2[j])
                j += 1
        while i < len(ar1):
            merged.append(ar1[i])
            i += 1
        while j < len(ar2):
            merged.append(ar2[j])
            j += 1
        total = len(merged)
        if total % 2 == 1:
            return merged[total // 2]
        return (merged[total // 2 - 1] + merged[total // 2]) / 2.0


def Test_Median_Different_Size():
    solution = Solution()

    class TestCase:
        def __init__(self, ar1, ar2, expected):
            self.ar1 = ar1
            self.ar2 = ar2
            self.expected = expected

    test_cases = [
        TestCase([900], [5, 8, 10, 20], 10),
        TestCase([-5, 3, 6, 12, 15], [-12, -10, -6, -3, 4, 10], 3),
        TestCase([1, 3], [2], 2),
        TestCase([1, 2], [3, 4], 2.5)
    ]

    for tc in test_cases:
        print(f"ar1: {tc.ar1}, ar2: {tc.ar2}, Expected: {tc.expected}")

        print("Binary Search:", solution.Median_Binary_Search_Optimal(tc.ar1, tc.ar2))
        print("Merge Count:", solution.Median_Merge_Count(tc.ar1, tc.ar2))
        print("Full Merge:", solution.Median_Full_Merge(tc.ar1, tc.ar2))

        print("-" * 50)


if __name__ == "__main__":
    Test_Median_Different_Size()
