"""
Problem: Largest Area Rectangular Submatrix with Equal 0s and 1s
URL: https://www.geeksforgeeks.org/largest-area-rectangular-sub-matrix-equal-number-1s-0s/

Problem Statement:
Given a binary matrix, find the largest rectangular sub-matrix with equal number of 1s and 0s. Replace 0 with -1, then find largest submatrix with sum 0.

Sample Input/Output:
Input: binary matrix
Output: largest area
"""


class Solution:
    def Largest_Rect_01_Kadane(self, matrix: list[list[int]]) -> int:
        """
        Kadane's Algorithm Approach
        Time Complexity: O(n^2*m)
        Space Complexity: O(m)
        """
        rows = len(matrix)
        cols = len(matrix[0])
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    matrix[i][j] = -1
        
        max_area = 0
        
        for top in range(rows):
            temp = [0] * cols
            
            for bottom in range(top, rows):
                for j in range(cols):
                    temp[j] += matrix[bottom][j]
                
                area = self._max_subarray_with_sum_zero(temp)
                max_area = max(max_area, area)
        
        return max_area
    
    def _max_subarray_with_sum_zero(self, arr: list[int]) -> int:
        prefix_sum = {}
        sum_val = 0
        max_len = 0
        
        for i in range(len(arr)):
            sum_val += arr[i]
            
            if sum_val == 0:
                max_len = i + 1
            
            if sum_val in prefix_sum:
                max_len = max(max_len, i - prefix_sum[sum_val])
            else:
                prefix_sum[sum_val] = i
        
        return max_len


def Test_LargestRectEqual01():
    solution = Solution()
    matrix = [
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    result = solution.Largest_Rect_01_Kadane(matrix)
    assert result >= 0


if __name__ == "__main__":
    Test_LargestRectEqual01()
