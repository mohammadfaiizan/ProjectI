"""
Problem: Maximum Sum Rectangle
URL: https://practice.geeksforgeeks.org/problems/maximum-sum-rectangle2948/1

Problem Statement:
Given a 2D matrix, find the maximum sum rectangle in it.

Sample Input/Output:
Input: 4x5 matrix
Output: maximum sum rectangle
"""


class Solution:
    def Max_Rect_Kadane(self, matrix: list[list[int]]) -> int:
        """
        Kadane's Algorithm for 2D
        Time Complexity: O(n^2*m)
        Space Complexity: O(m)
        """
        rows = len(matrix)
        cols = len(matrix[0])
        max_sum = float('-inf')
        
        for left in range(cols):
            temp = [0] * rows
            
            for right in range(left, cols):
                for i in range(rows):
                    temp[i] += matrix[i][right]
                
                current_sum = self._kadane(temp)
                max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def _kadane(self, arr: list[int]) -> int:
        max_sum = arr[0]
        current_sum = arr[0]
        
        for i in range(1, len(arr)):
            current_sum = max(arr[i], current_sum + arr[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum


def Test_MaxSumRectangle():
    solution = Solution()
    matrix = [
        [1, 2, -1, -4, -20],
        [-8, -3, 4, 2, 1],
        [3, 8, 10, 1, 3],
        [-4, -1, 1, 7, -6]
    ]
    result = solution.Max_Rect_Kadane(matrix)
    assert result > 0


if __name__ == "__main__":
    Test_MaxSumRectangle()
