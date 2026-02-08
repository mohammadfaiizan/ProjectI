"""
Problem: Common Elements in All Rows of a Matrix
URL: https://www.geeksforgeeks.org/common-elements-in-all-rows-of-a-given-matrix/

Problem Statement:
Given an M x N matrix, find all common elements present in all rows.

Sample Input/Output:
Input: mat = [[1, 2, 1, 4, 8],
              [3, 7, 8, 5, 1],
              [8, 7, 7, 3, 1],
              [8, 1, 2, 7, 9]]
Output: [1, 8]
Explanation: 1 and 8 appear in all rows.

Input: mat = [[1, 2, 3],
              [4, 5, 6]]
Output: []
Explanation: No common element in all rows.
"""


class Solution:
    def Common_Elements_Map_Optimal(self, mat):
        """
        Row-wise Map Counting - Track elements appearing in each row
        Time Complexity: O(m * n)
        Space Complexity: O(n)
        """
        rows, cols = len(mat), len(mat[0])
        mp = {}
        for j in range(cols):
            mp[mat[0][j]] = 1
        for i in range(1, rows):
            for j in range(cols):
                if mat[i][j] in mp and mp[mat[i][j]] == i:
                    mp[mat[i][j]] = i + 1
        result = []
        for val, count in mp.items():
            if count == rows:
                result.append(val)
        return sorted(result)

    def Common_Elements_Set_Intersection(self, mat):
        """
        Set Intersection - Intersect sets of each row
        Time Complexity: O(m * n * log n)
        Space Complexity: O(n)
        """
        common = set(mat[0])
        for i in range(1, len(mat)):
            row_set = set(mat[i])
            common = common.intersection(row_set)
        return sorted(list(common))


def Test_Common_Elements_All_Rows():
    solution = Solution()

    test_cases = [
        [[1, 2, 1, 4, 8], [3, 7, 8, 5, 1], [8, 7, 7, 3, 1], [8, 1, 2, 7, 9]],
        [[1, 2, 3], [4, 5, 6]],
        [[5, 3, 7], [5, 7, 3], [5, 3, 7]]
    ]

    for mat in test_cases:
        print("Matrix:")
        for row in mat:
            print(" ".join(str(x) for x in row))

        r1 = solution.Common_Elements_Map_Optimal(mat)
        print("Map:", " ".join(str(x) for x in r1) if r1 else "(none)")

        r2 = solution.Common_Elements_Set_Intersection(mat)
        print("Set Intersection:", " ".join(str(x) for x in r2) if r2 else "(none)")

        print("-" * 50)


if __name__ == "__main__":
    Test_Common_Elements_All_Rows()
