"""
Problem: Kth Smallest Element in a Row-Column Sorted Matrix
URL: https://practice.geeksforgeeks.org/problems/kth-element-in-matrix/1

Problem Statement:
Given an N x N matrix where every row and column is sorted in non-decreasing order,
find the kth smallest element in the matrix.

Sample Input/Output:
Input: matrix = [[10, 20, 30, 40],
                 [15, 25, 35, 45],
                 [25, 29, 37, 48],
                 [32, 33, 39, 50]], K = 7
Output: 30
Explanation: Sorted elements: 10,15,20,25,25,29,30,... 7th is 30.

Input: matrix = [[16, 28, 60, 64],
                 [22, 41, 63, 91],
                 [27, 50, 87, 93],
                 [36, 78, 87, 94]], K = 3
Output: 27
"""

import heapq


class Solution:
    def Kth_Smallest_Min_Heap_Optimal(self, mat, k):
        """
        Min Heap - Push first row/col, extract min k times
        Time Complexity: O(n + k * log n)
        Space Complexity: O(n)
        """
        n = len(mat)
        pq = []
        for i in range(n):
            heapq.heappush(pq, (mat[i][0], i, 0))
        for _ in range(1, k):
            val, r, c = heapq.heappop(pq)
            if c + 1 < n:
                heapq.heappush(pq, (mat[r][c + 1], r, c + 1))
        return pq[0][0]

    def Kth_Smallest_Binary_Search(self, mat, k):
        """
        Binary Search on Value - Count elements less than or equal to mid
        Time Complexity: O(n * log(max - min))
        Space Complexity: O(1)
        """
        n = len(mat)
        lo, hi = mat[0][0], mat[n - 1][n - 1]
        while lo < hi:
            mid = lo + (hi - lo) // 2
            count = self.Count_Less_Equal(mat, mid, n)
            if count < k:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def Kth_Smallest_Flatten(self, mat, k):
        """
        Flatten and Sort - Put all elements in array and sort
        Time Complexity: O(n^2 * log(n^2))
        Space Complexity: O(n^2)
        """
        all_elements = []
        for row in mat:
            all_elements.extend(row)
        all_elements.sort()
        return all_elements[k - 1]

    def Count_Less_Equal(self, mat, mid, n):
        count = 0
        j = n - 1
        for i in range(n):
            while j >= 0 and mat[i][j] > mid:
                j -= 1
            count += j + 1
        return count


def Test_Kth_Smallest_Matrix():
    solution = Solution()

    class TestCase:
        def __init__(self, mat, k, expected):
            self.mat = mat
            self.k = k
            self.expected = expected

    test_cases = [
        TestCase([[10, 20, 30, 40], [15, 25, 35, 45], [25, 29, 37, 48], [32, 33, 39, 50]], 7, 30),
        TestCase([[16, 28, 60, 64], [22, 41, 63, 91], [27, 50, 87, 93], [36, 78, 87, 94]], 3, 27),
        TestCase([[1, 5, 9], [10, 11, 13], [12, 13, 15]], 8, 13)
    ]

    for tc in test_cases:
        print(f"K={tc.k}, Expected={tc.expected}")

        print("Min Heap:", solution.Kth_Smallest_Min_Heap_Optimal(tc.mat, tc.k))
        print("Binary Search:", solution.Kth_Smallest_Binary_Search(tc.mat, tc.k))
        print("Flatten:", solution.Kth_Smallest_Flatten(tc.mat, tc.k))

        print("-" * 50)


if __name__ == "__main__":
    Test_Kth_Smallest_Matrix()
