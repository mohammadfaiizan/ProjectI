"""
Problem: Rasta and Kheshtak
URL: https://www.hackerearth.com/practice/algorithms/searching/binary-search/practice-problems/algorithm/rasta-and-kheshtak/

Problem Statement:
Given two binary matrices, find the largest square submatrix of matrix1
that appears in matrix2.

Sample Input:
3 3
1 0 1
0 1 0
1 0 1
4 4
1 0 1 0
0 1 0 1
1 0 1 0
0 1 0 1

Sample Output:
2
"""


class Solution:
    def Solve_2D_Hashing_Binary_Search(self, matrix1, matrix2):
        """
        Approach: Use 2D hashing to compute hash values for all submatrices,
        then binary search on the answer (size of square) to find maximum size.
        Time Complexity: O(n^2 * m^2 * log(min(n,m))) where n,m are matrix dimensions
        Space Complexity: O(n^2 * m^2) for hash storage
        """
        n1, m1 = len(matrix1), len(matrix1[0])
        n2, m2 = len(matrix2), len(matrix2[0])
        
        max_size = 0
        low = 1
        high = min(min(n1, m1), min(n2, m2))
        
        while low <= high:
            mid = (low + high) // 2
            if self.Has_Submatrix_Of_Size(matrix1, matrix2, mid):
                max_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return max_size
    
    def Has_Submatrix_Of_Size(self, mat1, mat2, size):
        n1, m1 = len(mat1), len(mat1[0])
        n2, m2 = len(mat2), len(mat2[0])
        
        submatrices = set()
        
        for i in range(n1 - size + 1):
            for j in range(m1 - size + 1):
                sub = tuple(tuple(mat1[i + x][j + y] for y in range(size)) for x in range(size))
                submatrices.add(sub)
        
        for i in range(n2 - size + 1):
            for j in range(m2 - size + 1):
                sub = tuple(tuple(mat2[i + x][j + y] for y in range(size)) for x in range(size))
                if sub in submatrices:
                    return True
        
        return False


def Test_Rasta_And_Kheshtak():
    sol = Solution()
    
    matrix1 = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ]
    matrix2 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    result1 = sol.Solve_2D_Hashing_Binary_Search(matrix1, matrix2)
    assert result1 == 2
    
    matrix3 = [
        [1, 1],
        [1, 1]
    ]
    matrix4 = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ]
    result2 = sol.Solve_2D_Hashing_Binary_Search(matrix3, matrix4)
    assert result2 == 2
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Rasta_And_Kheshtak()
