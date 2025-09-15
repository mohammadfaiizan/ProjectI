"""
Problem: Maximum Sum Submatrix No Larger Than K
URL: https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/description/

Problem Statement:
Given an m x n matrix matrix and an integer k, return the max sum of a rectangle in the matrix such that its sum is no larger than k.
It is guaranteed that there will be a rectangle with a sum no larger than k.

Sample Input/Output:
Input: matrix = [[1,0,1],[0,-2,3]], k = 2
Output: 2
Explanation: Because the sum of the blue rectangle [[0, 1], [-2, 3]] is 2, and 2 is the max number no larger than k (k = 2).

Input: matrix = [[2,2,-1]], k = 3
Output: 3
Explanation: Because the sum of the blue rectangle [[2, 2, -1]] is 3, and 3 is the max number no larger than k (k = 3).
"""

from typing import List, Tuple
import bisect

class Solution:
    def Max_Sum_Submatrix_K_Brute_Force(self, matrix: List[List[int]], k: int) -> int:
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
                        
                        if current_sum <= k:
                            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def Max_Sum_Submatrix_K_Prefix_Sum(self, matrix: List[List[int]], k: int) -> int:
        """
        Prefix Sum - Use prefix sum for O(1) rectangle sum calculation
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
                        
                        if rectangle_sum <= k:
                            max_sum = max(max_sum, rectangle_sum)
        
        return max_sum
    
    def Max_Sum_Submatrix_K_1D_Optimization_Optimal(self, matrix: List[List[int]], k: int) -> int:
        """
        1D Optimization Optimal - Reduce to 1D problem with binary search
        Time Complexity: O(min(m,n)² * max(m,n) * log(max(m,n)))
        Space Complexity: O(max(m,n))
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        
        if m > n:
            matrix = list(zip(*matrix))
            m, n = n, m
        
        def Max_Subarray_Sum_No_Larger_Than_K(arr: List[int], k: int) -> int:
            max_sum = float('-inf')
            prefix_sums = [0]
            current_sum = 0
            
            for num in arr:
                current_sum += num
                
                target = current_sum - k
                idx = bisect.bisect_left(prefix_sums, target)
                
                if idx < len(prefix_sums):
                    max_sum = max(max_sum, current_sum - prefix_sums[idx])
                
                bisect.insort(prefix_sums, current_sum)
            
            return max_sum
        
        max_sum = float('-inf')
        
        for top in range(m):
            temp = [0] * n
            
            for bottom in range(top, m):
                for j in range(n):
                    temp[j] += matrix[bottom][j]
                
                current_max = Max_Subarray_Sum_No_Larger_Than_K(temp, k)
                max_sum = max(max_sum, current_max)
        
        return max_sum
    
    def Max_Sum_Submatrix_K_TreeSet_Alternative(self, matrix: List[List[int]], k: int) -> int:
        """
        TreeSet Alternative - Using sorted list for efficient operations
        Time Complexity: O(min(m,n)² * max(m,n) * log(max(m,n)))
        Space Complexity: O(max(m,n))
        """
        if not matrix or not matrix[0]:
            return 0
        
        from sortedcontainers import SortedList
        
        m, n = len(matrix), len(matrix[0])
        
        if m > n:
            matrix = list(zip(*matrix))
            m, n = n, m
        
        def Max_Subarray_Sum_No_Larger_Than_K_TreeSet(arr: List[int], k: int) -> int:
            max_sum = float('-inf')
            sorted_prefix = SortedList([0])
            current_sum = 0
            
            for num in arr:
                current_sum += num
                
                target = current_sum - k
                idx = sorted_prefix.bisect_left(target)
                
                if idx < len(sorted_prefix):
                    max_sum = max(max_sum, current_sum - sorted_prefix[idx])
                
                sorted_prefix.add(current_sum)
            
            return max_sum
        
        max_sum = float('-inf')
        
        for top in range(m):
            temp = [0] * n
            
            for bottom in range(top, m):
                for j in range(n):
                    temp[j] += matrix[bottom][j]
                
                current_max = Max_Subarray_Sum_No_Larger_Than_K_TreeSet(temp, k)
                max_sum = max(max_sum, current_max)
        
        return max_sum
    
    def Max_Sum_Submatrix_K_With_Coordinates(self, matrix: List[List[int]], k: int) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
        """
        With Coordinates - Return max sum and rectangle coordinates
        Time Complexity: O(m²*n²)
        Space Complexity: O(m*n)
        """
        if not matrix or not matrix[0]:
            return 0, (-1, -1), (-1, -1)
        
        m, n = len(matrix), len(matrix[0])
        
        prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
        
        max_sum = float('-inf')
        best_coords = ((-1, -1), (-1, -1))
        
        for r1 in range(m):
            for c1 in range(n):
                for r2 in range(r1, m):
                    for c2 in range(c1, n):
                        rectangle_sum = (prefix[r2+1][c2+1] - prefix[r1][c2+1] - 
                                       prefix[r2+1][c1] + prefix[r1][c1])
                        
                        if rectangle_sum <= k and rectangle_sum > max_sum:
                            max_sum = rectangle_sum
                            best_coords = ((r1, c1), (r2, c2))
        
        return max_sum, best_coords[0], best_coords[1]
    
    def Max_Sum_Submatrix_K_DP_Approach(self, matrix: List[List[int]], k: int) -> int:
        """
        DP Approach - Dynamic programming with constraint checking
        Time Complexity: O(m²*n²)
        Space Complexity: O(m*n)
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
                
                current_sum = 0
                for start in range(n):
                    current_sum = 0
                    for end in range(start, n):
                        current_sum += temp[end]
                        if current_sum <= k:
                            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def Max_Sum_Submatrix_K_Sliding_Window(self, matrix: List[List[int]], k: int) -> int:
        """
        Sliding Window - Use sliding window for 1D subarray problem
        Time Complexity: O(m²*n²)
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
                
                for i in range(n):
                    current_sum = 0
                    for j in range(i, n):
                        current_sum += temp[j]
                        if current_sum <= k:
                            max_sum = max(max_sum, current_sum)
                        if current_sum > k:
                            break
        
        return max_sum
    
    def Max_Sum_Submatrix_K_All_Valid_Rectangles(self, matrix: List[List[int]], k: int) -> Tuple[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        All Valid Rectangles - Find all rectangles with max sum <= k
        Time Complexity: O(m²*n²)
        Space Complexity: O(m²*n²)
        """
        if not matrix or not matrix[0]:
            return 0, []
        
        m, n = len(matrix), len(matrix[0])
        
        prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
        
        max_sum = float('-inf')
        max_rectangles = []
        
        for r1 in range(m):
            for c1 in range(n):
                for r2 in range(r1, m):
                    for c2 in range(c1, n):
                        rectangle_sum = (prefix[r2+1][c2+1] - prefix[r1][c2+1] - 
                                       prefix[r2+1][c1] + prefix[r1][c1])
                        
                        if rectangle_sum <= k:
                            if rectangle_sum > max_sum:
                                max_sum = rectangle_sum
                                max_rectangles = [((r1, c1), (r2, c2))]
                            elif rectangle_sum == max_sum:
                                max_rectangles.append(((r1, c1), (r2, c2)))
        
        return max_sum, max_rectangles

def Test_Max_Sum_Submatrix_K():
    solution = Solution()
    
    test_cases = [
        ([[1,0,1],[0,-2,3]], 2, 2),
        ([[2,2,-1]], 3, 3),
        ([[5,-4,-3,4],[-3,-4,4,5],[5,1,5,-4]], 3, 2),
        ([[1,1],[1,1]], 4, 4),
        ([[-1,-2],[-3,-4]], -1, -1)
    ]
    
    methods = [
        ("Prefix Sum", solution.Max_Sum_Submatrix_K_Prefix_Sum),
        ("1D Optimization Optimal", solution.Max_Sum_Submatrix_K_1D_Optimization_Optimal),
        ("DP Approach", solution.Max_Sum_Submatrix_K_DP_Approach),
        ("Sliding Window", solution.Max_Sum_Submatrix_K_Sliding_Window)
    ]
    
    for matrix, k, expected in test_cases:
        print(f"Matrix: {matrix}, k: {k}")
        print(f"Expected: {expected}")
        
        if len(matrix) <= 3 and len(matrix[0]) <= 3:
            result_bf = solution.Max_Sum_Submatrix_K_Brute_Force([row[:] for row in matrix], k)
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method([row[:] for row in matrix], k)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_sum, top_left, bottom_right = solution.Max_Sum_Submatrix_K_With_Coordinates([row[:] for row in matrix], k)
        print(f"With Coordinates: Sum={max_sum}, Top-left={top_left}, Bottom-right={bottom_right}")
        
        if len(matrix) <= 3:
            max_sum, all_rectangles = solution.Max_Sum_Submatrix_K_All_Valid_Rectangles([row[:] for row in matrix], k)
            print(f"All Valid Rectangles: Sum={max_sum}, Count={len(all_rectangles)}")
            for rect in all_rectangles:
                print(f"  {rect}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Sum_Submatrix_K()
