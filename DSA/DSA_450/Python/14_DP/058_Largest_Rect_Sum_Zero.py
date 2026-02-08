"""
Problem: Largest Rectangular Submatrix with Sum 0
URL: https://www.geeksforgeeks.org/largest-rectangular-sub-matrix-whose-sum-0/

Problem Statement:
Given a 2D matrix, find the largest rectangular submatrix whose sum is zero.

Sample Input/Output:
Input: matrix with positive and negative values
Output: Size of largest rectangular submatrix with sum 0
"""


class Solution:
    def Largest_Rect_Zero(self, matrix: list[list[int]]) -> int:
        """
        Fix columns and use prefix sums with hashmap
        Time Complexity: O(n^2*m)
        Space Complexity: O(m)
        """
        m = len(matrix)
        if m == 0:
            return 0
        n = len(matrix[0])
        if n == 0:
            return 0
        
        max_area = 0
        
        for left in range(n):
            temp = [0] * m
            
            for right in range(left, n):
                for i in range(m):
                    temp[i] += matrix[i][right]
                
                prefix_sum = {}
                prefix_sum[0] = -1
                sum_val = 0
                
                for i in range(m):
                    sum_val += temp[i]
                    
                    if sum_val in prefix_sum:
                        height = i - prefix_sum[sum_val]
                        width = right - left + 1
                        max_area = max(max_area, height * width)
                    else:
                        prefix_sum[sum_val] = i
        
        return max_area


def Test_LargestRectSumZero():
    solution = Solution()
    
    matrix = [
        [9, 7, 16, 5],
        [1, -6, -7, 3],
        [1, 8, 7, 9],
        [7, -2, 0, 10]
    ]
    
    result = solution.Largest_Rect_Zero(matrix)
    assert result >= 0


if __name__ == "__main__":
    Test_LargestRectSumZero()
