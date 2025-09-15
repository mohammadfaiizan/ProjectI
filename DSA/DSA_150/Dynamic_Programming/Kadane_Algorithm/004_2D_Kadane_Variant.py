"""
Problem: 2D Kadane's Algorithm Variant (Maximum Sum Rectangle)
URL: Similar to https://www.geeksforgeeks.org/maximum-sum-rectangle-in-a-2d-matrix-dp-27/

Problem Statement:
Given a 2D matrix, find the maximum sum of a rectangular submatrix.
This is a 2D extension of Kadane's algorithm for maximum subarray sum.

Sample Input/Output:
Input: matrix = [[1,2,-1,-4,-20],[-8,-3,4,2,1],[3,8,10,1,3],[-4,-1,1,7,-6]]
Output: 29
Explanation: The rectangle from (1,1) to (3,3) has the maximum sum of 29.

Input: matrix = [[-1,-2],[-3,-4]]
Output: -1
Explanation: The maximum sum rectangle is the single element -1.
"""

from typing import List, Tuple

class Solution:
    def Max_Sum_Rectangle_Brute_Force(self, matrix: List[List[int]]) -> int:
        """
        Brute Force - Check all possible rectangles
        Time Complexity: O(m²*n²*m*n)
        Space Complexity: O(1)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        max_sum = float('-inf')
        
        for r1 in range(m):
            for c1 in range(n):
                for r2 in range(r1, m):
                    for c2 in range(c1, n):
                        current_sum = 0
                        for i in range(r1, r2 + 1):
                            for j in range(c1, c2 + 1):
                                current_sum += matrix[i][j]
                        max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def Max_Sum_Rectangle_Prefix_Sum(self, matrix: List[List[int]]) -> int:
        """
        Prefix Sum - Use prefix sum for rectangle sum calculation
        Time Complexity: O(m²*n²)
        Space Complexity: O(m*n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        
        prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
        
        max_sum = float('-inf')
        
        for r1 in range(m):
            for c1 in range(n):
                for r2 in range(r1, m):
                    for c2 in range(c1, n):
                        rectangle_sum = (prefix[r2+1][c2+1] - prefix[r1][c2+1] - 
                                       prefix[r2+1][c1] + prefix[r1][c1])
                        max_sum = max(max_sum, rectangle_sum)
        
        return max_sum
    
    def Max_Sum_Rectangle_2D_Kadane_Optimal(self, matrix: List[List[int]]) -> int:
        """
        2D Kadane Optimal - Reduce to 1D Kadane for each row pair
        Time Complexity: O(m²*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        def Kadane_1D(arr: List[int]) -> int:
            max_ending_here = max_so_far = arr[0]
            for i in range(1, len(arr)):
                max_ending_here = max(arr[i], max_ending_here + arr[i])
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        m, n = len(matrix), len(matrix[0])
        max_sum = float('-inf')
        
        for top in range(m):
            temp = [0] * n
            
            for bottom in range(top, m):
                for j in range(n):
                    temp[j] += matrix[bottom][j]
                
                current_max = Kadane_1D(temp)
                max_sum = max(max_sum, current_max)
        
        return max_sum
    
    def Max_Sum_Rectangle_With_Coordinates(self, matrix: List[List[int]]) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
        """
        With Coordinates - Return max sum and rectangle coordinates
        Time Complexity: O(m²*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0, (-1, -1), (-1, -1)
        
        def Kadane_1D_With_Indices(arr: List[int]) -> Tuple[int, int, int]:
            max_ending_here = max_so_far = arr[0]
            start = end = 0
            temp_start = 0
            
            for i in range(1, len(arr)):
                if max_ending_here < 0:
                    max_ending_here = arr[i]
                    temp_start = i
                else:
                    max_ending_here += arr[i]
                
                if max_ending_here > max_so_far:
                    max_so_far = max_ending_here
                    start = temp_start
                    end = i
            
            return max_so_far, start, end
        
        m, n = len(matrix), len(matrix[0])
        max_sum = float('-inf')
        final_top = final_bottom = final_left = final_right = 0
        
        for top in range(m):
            temp = [0] * n
            
            for bottom in range(top, m):
                for j in range(n):
                    temp[j] += matrix[bottom][j]
                
                current_max, left, right = Kadane_1D_With_Indices(temp)
                
                if current_max > max_sum:
                    max_sum = current_max
                    final_top = top
                    final_bottom = bottom
                    final_left = left
                    final_right = right
        
        return max_sum, (final_top, final_left), (final_bottom, final_right)
    
    def Max_Sum_Rectangle_DP_2D(self, matrix: List[List[int]]) -> int:
        """
        DP 2D - Dynamic programming approach for 2D matrix
        Time Complexity: O(m²*n)
        Space Complexity: O(m*n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        
        dp = [[[float('-inf')] * n for _ in range(m)] for _ in range(m)]
        
        for top in range(m):
            for bottom in range(top, m):
                for col in range(n):
                    column_sum = sum(matrix[row][col] for row in range(top, bottom + 1))
                    
                    if col == 0:
                        dp[top][bottom][col] = column_sum
                    else:
                        dp[top][bottom][col] = max(column_sum, dp[top][bottom][col-1] + column_sum)
        
        max_sum = float('-inf')
        for top in range(m):
            for bottom in range(top, m):
                for col in range(n):
                    max_sum = max(max_sum, dp[top][bottom][col])
        
        return max_sum
    
    def Max_Sum_Rectangle_Divide_Conquer(self, matrix: List[List[int]]) -> int:
        """
        Divide Conquer - Divide and conquer approach
        Time Complexity: O(m²*n*log(n))
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        def Max_Crossing_Sum(arr: List[int], left: int, mid: int, right: int) -> int:
            left_sum = float('-inf')
            total = 0
            for i in range(mid, left - 1, -1):
                total += arr[i]
                left_sum = max(left_sum, total)
            
            right_sum = float('-inf')
            total = 0
            for i in range(mid + 1, right + 1):
                total += arr[i]
                right_sum = max(right_sum, total)
            
            return left_sum + right_sum
        
        def Max_Subarray_Divide_Conquer(arr: List[int], left: int, right: int) -> int:
            if left == right:
                return arr[left]
            
            mid = (left + right) // 2
            
            left_sum = Max_Subarray_Divide_Conquer(arr, left, mid)
            right_sum = Max_Subarray_Divide_Conquer(arr, mid + 1, right)
            cross_sum = Max_Crossing_Sum(arr, left, mid, right)
            
            return max(left_sum, right_sum, cross_sum)
        
        m, n = len(matrix), len(matrix[0])
        max_sum = float('-inf')
        
        for top in range(m):
            temp = [0] * n
            
            for bottom in range(top, m):
                for j in range(n):
                    temp[j] += matrix[bottom][j]
                
                current_max = Max_Subarray_Divide_Conquer(temp, 0, n - 1)
                max_sum = max(max_sum, current_max)
        
        return max_sum
    
    def Max_Sum_Rectangle_All_Rectangles(self, matrix: List[List[int]]) -> Tuple[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        All Rectangles - Find all rectangles with maximum sum
        Time Complexity: O(m²*n²)
        Space Complexity: O(m²*n²)
        """
        max_sum = self.Max_Sum_Rectangle_2D_Kadane_Optimal(matrix)
        max_rectangles = []
        
        if not matrix or not matrix[0]:
            return 0, []
        
        m, n = len(matrix), len(matrix[0])
        
        prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
        
        for r1 in range(m):
            for c1 in range(n):
                for r2 in range(r1, m):
                    for c2 in range(c1, n):
                        rectangle_sum = (prefix[r2+1][c2+1] - prefix[r1][c2+1] - 
                                       prefix[r2+1][c1] + prefix[r1][c1])
                        
                        if rectangle_sum == max_sum:
                            max_rectangles.append(((r1, c1), (r2, c2)))
        
        return max_sum, max_rectangles
    
    def Max_Sum_Rectangle_Space_Optimized(self, matrix: List[List[int]]) -> int:
        """
        Space Optimized - Optimize space usage
        Time Complexity: O(m²*n)
        Space Complexity: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        max_sum = float('-inf')
        
        for top in range(m):
            temp = [0] * n
            
            for bottom in range(top, m):
                for j in range(n):
                    temp[j] += matrix[bottom][j]
                
                max_ending_here = max_so_far = temp[0]
                
                for i in range(1, n):
                    max_ending_here = max(temp[i], max_ending_here + temp[i])
                    max_so_far = max(max_so_far, max_ending_here)
                
                max_sum = max(max_sum, max_so_far)
        
        return max_sum

def Test_Max_Sum_Rectangle():
    solution = Solution()
    
    test_cases = [
        ([[1,2,-1,-4,-20],[-8,-3,4,2,1],[3,8,10,1,3],[-4,-1,1,7,-6]], 29),
        ([[-1,-2],[-3,-4]], -1),
        ([[1,2,3],[4,5,6],[7,8,9]], 45),
        ([[-1,2,-3],[4,-5,6],[-7,8,-9]], 8),
        ([[1]], 1),
        ([[1,2],[3,4]], 10)
    ]
    
    methods = [
        ("Prefix Sum", solution.Max_Sum_Rectangle_Prefix_Sum),
        ("2D Kadane Optimal", solution.Max_Sum_Rectangle_2D_Kadane_Optimal),
        ("DP 2D", solution.Max_Sum_Rectangle_DP_2D),
        ("Divide Conquer", solution.Max_Sum_Rectangle_Divide_Conquer),
        ("Space Optimized", solution.Max_Sum_Rectangle_Space_Optimized)
    ]
    
    for matrix, expected in test_cases:
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}")
        
        if len(matrix) <= 3 and len(matrix[0]) <= 3:
            result_bf = solution.Max_Sum_Rectangle_Brute_Force([row[:] for row in matrix])
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method([row[:] for row in matrix])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_sum, top_left, bottom_right = solution.Max_Sum_Rectangle_With_Coordinates([row[:] for row in matrix])
        print(f"With Coordinates: Sum={max_sum}, Top-left={top_left}, Bottom-right={bottom_right}")
        
        if len(matrix) <= 3 and len(matrix[0]) <= 4:
            max_sum, all_rectangles = solution.Max_Sum_Rectangle_All_Rectangles([row[:] for row in matrix])
            print(f"All Max Rectangles: Sum={max_sum}, Count={len(all_rectangles)}")
            for rect in all_rectangles[:3]:
                print(f"  {rect}")
            if len(all_rectangles) > 3:
                print(f"  ... and {len(all_rectangles) - 3} more")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Sum_Rectangle()
