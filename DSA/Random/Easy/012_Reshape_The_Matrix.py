"""
Problem: Reshape the Matrix
URL: https://leetcode.com/problems/reshape-the-matrix/

Problem Statement:
You are given an m x n matrix mat and two integers r and c representing the number of 
rows and the number of columns of the wanted reshaped matrix.

The reshaped matrix should be filled with all the elements of the original matrix in 
the same row-traversing order as they were.

If the reshape operation is possible and legal, output the new reshaped matrix; 
otherwise, output the original matrix.

Sample Input/Output:
Input: mat = [[1,2],[3,4]], r = 1, c = 4
Output: [[1,2,3,4]]

Input: mat = [[1,2],[3,4]], r = 2, c = 4
Output: [[1,2],[3,4]]
Explanation: Not possible to reshape

Input: mat = [[1,2,3,4]], r = 2, c = 2
Output: [[1,2],[3,4]]
"""

from typing import List

class Solution:
    def Reshape_Matrix_Brute_Force(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        """
        Brute Force Approach - Flatten and rebuild
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(mat), len(mat[0])
        
        if m * n != r * c:
            return mat
        
        flat = []
        for row in mat:
            for num in row:
                flat.append(num)
        
        result = []
        idx = 0
        for i in range(r):
            row = []
            for j in range(c):
                row.append(flat[idx])
                idx += 1
            result.append(row)
        
        return result
    
    def Reshape_Matrix_Single_Pass(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        """
        Single Pass Approach
        Time Complexity: O(m * n)
        Space Complexity: O(1) excluding output
        """
        m, n = len(mat), len(mat[0])
        
        if m * n != r * c:
            return mat
        
        result = [[0] * c for _ in range(r)]
        row_idx, col_idx = 0, 0
        
        for i in range(m):
            for j in range(n):
                result[row_idx][col_idx] = mat[i][j]
                col_idx += 1
                if col_idx == c:
                    row_idx += 1
                    col_idx = 0
        
        return result
    
    def Reshape_Matrix_Index_Mapping(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        """
        Index Mapping Approach - Optimal solution
        Time Complexity: O(m * n)
        Space Complexity: O(1) excluding output
        """
        m, n = len(mat), len(mat[0])
        
        if m * n != r * c:
            return mat
        
        result = [[0] * c for _ in range(r)]
        
        for i in range(m * n):
            result[i // c][i % c] = mat[i // n][i % n]
        
        return result
    
    def Reshape_Matrix_List_Comprehension(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        """
        List Comprehension Approach
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(mat), len(mat[0])
        
        if m * n != r * c:
            return mat
        
        flat = [num for row in mat for num in row]
        
        return [flat[i * c:(i + 1) * c] for i in range(r)]
    
    def Reshape_Matrix_Generator(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        """
        Generator Approach
        Time Complexity: O(m * n)
        Space Complexity: O(1) excluding output
        """
        m, n = len(mat), len(mat[0])
        
        if m * n != r * c:
            return mat
        
        def Generate_Elements():
            for row in mat:
                for num in row:
                    yield num
        
        gen = Generate_Elements()
        result = []
        
        for _ in range(r):
            row = [next(gen) for _ in range(c)]
            result.append(row)
        
        return result

def Test_Reshape_Matrix():
    solution = Solution()
    
    test_cases = [
        ([[1,2],[3,4]], 1, 4, [[1,2,3,4]]),
        ([[1,2],[3,4]], 2, 4, [[1,2],[3,4]]),
        ([[1,2,3,4]], 2, 2, [[1,2],[3,4]]),
        ([[1]], 1, 1, [[1]]),
        ([[1,2,3],[4,5,6]], 2, 3, [[1,2,3],[4,5,6]])
    ]
    
    for mat, r, c, expected in test_cases:
        result1 = solution.Reshape_Matrix_Brute_Force([row[:] for row in mat], r, c)
        result2 = solution.Reshape_Matrix_Single_Pass([row[:] for row in mat], r, c)
        result3 = solution.Reshape_Matrix_Index_Mapping([row[:] for row in mat], r, c)
        result4 = solution.Reshape_Matrix_List_Comprehension([row[:] for row in mat], r, c)
        result5 = solution.Reshape_Matrix_Generator([row[:] for row in mat], r, c)
        
        print(f"Matrix: {mat}, r: {r}, c: {c}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Single Pass: {result2}")
        print(f"Index Mapping: {result3}")
        print(f"List Comprehension: {result4}")
        print(f"Generator: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Reshape_Matrix()

