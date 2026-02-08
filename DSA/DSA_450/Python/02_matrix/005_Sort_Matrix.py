"""
Problem: Sort the Given Matrix
URL: https://practice.geeksforgeeks.org/problems/sorted-matrix2333/1

Problem Statement:
Given an N x N matrix, sort all elements of the matrix in increasing order
and put them back into the matrix in row-wise fashion.

Sample Input/Output:
Input: matrix = [[10, 20, 30, 40],
                 [15, 25, 35, 45],
                 [27, 29, 37, 48],
                 [32, 33, 39, 50]]
Output: [[10, 15, 20, 25],
         [27, 29, 30, 32],
         [33, 35, 37, 39],
         [40, 45, 48, 50]]

Input: matrix = [[5, 4], [3, 1]]
Output: [[1, 3], [4, 5]]
"""

import heapq


class Solution:
    def Sort_Matrix_Flatten_Optimal(self, mat):
        """
        Flatten, Sort, Refill - Extract all elements, sort, put back
        Time Complexity: O(n^2 * log(n^2))
        Space Complexity: O(n^2)
        """
        n = len(mat)
        temp = []
        for i in range(n):
            for j in range(n):
                temp.append(mat[i][j])
        temp.sort()
        k = 0
        for i in range(n):
            for j in range(n):
                mat[i][j] = temp[k]
                k += 1
        return mat

    def Sort_Matrix_Min_Heap(self, mat):
        """
        Min Heap - Use priority queue for sorted extraction
        Time Complexity: O(n^2 * log(n^2))
        Space Complexity: O(n^2)
        """
        n = len(mat)
        pq = []
        for i in range(n):
            for j in range(n):
                heapq.heappush(pq, mat[i][j])
        for i in range(n):
            for j in range(n):
                mat[i][j] = heapq.heappop(pq)
        return mat


def Test_Sort_Matrix():
    solution = Solution()

    test_cases = [
        [[10, 20, 30, 40], [15, 25, 35, 45], [27, 29, 37, 48], [32, 33, 39, 50]],
        [[5, 4], [3, 1]],
        [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    ]

    for mat in test_cases:
        print("Original:")
        for row in mat:
            print("\t".join(str(x) for x in row))

        import copy
        r1 = solution.Sort_Matrix_Flatten_Optimal(copy.deepcopy(mat))
        print("Flatten Sort:")
        for row in r1:
            print("\t".join(str(x) for x in row))

        r2 = solution.Sort_Matrix_Min_Heap(copy.deepcopy(mat))
        print("Min Heap:")
        for row in r2:
            print("\t".join(str(x) for x in row))

        print("-" * 50)


if __name__ == "__main__":
    Test_Sort_Matrix()
