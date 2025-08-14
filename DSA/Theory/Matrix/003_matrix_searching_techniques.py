"""
Matrix Searching Techniques
===========================

Topics: Linear search, binary search, sorted matrix search
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Easy to Hard
"""

from typing import List, Optional, Tuple

class MatrixSearchingTechniques:
    
    # ==========================================
    # 1. BASIC SEARCHING
    # ==========================================
    
    def linear_search(self, matrix: List[List[int]], target: int) -> Optional[Tuple[int, int]]:
        """Linear search in matrix"""
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == target:
                    return (i, j)
        return None
    
    # ==========================================
    # 2. SORTED MATRIX SEARCHING
    # ==========================================
    
    def search_sorted_matrix(self, matrix: List[List[int]], target: int) -> bool:
        """LC 74: Search in sorted matrix (row and column sorted)"""
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1
        
        while row < m and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False
    
    def search_row_col_sorted(self, matrix: List[List[int]], target: int) -> bool:
        """LC 240: Search 2D Matrix II"""
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1
        
        while row < m and col >= 0:
            current = matrix[row][col]
            if current == target:
                return True
            elif current > target:
                col -= 1
            else:
                row += 1
        
        return False
    
    def binary_search_matrix(self, matrix: List[List[int]], target: int) -> bool:
        """Binary search treating matrix as 1D array"""
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1
        
        while left <= right:
            mid = (left + right) // 2
            mid_val = matrix[mid // n][mid % n]
            
            if mid_val == target:
                return True
            elif mid_val < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return False
    
    # ==========================================
    # 3. ADVANCED SEARCHING
    # ==========================================
    
    def find_peak_element_2d(self, matrix: List[List[int]]) -> List[int]:
        """LC 1901: Find Peak Element II"""
        m, n = len(matrix), len(matrix[0])
        
        def find_max_in_col(col):
            max_row = 0
            for i in range(1, m):
                if matrix[i][col] > matrix[max_row][col]:
                    max_row = i
            return max_row
        
        left, right = 0, n - 1
        
        while left <= right:
            mid_col = (left + right) // 2
            max_row = find_max_in_col(mid_col)
            
            left_val = matrix[max_row][mid_col - 1] if mid_col > 0 else -1
            right_val = matrix[max_row][mid_col + 1] if mid_col < n - 1 else -1
            
            if matrix[max_row][mid_col] > left_val and matrix[max_row][mid_col] > right_val:
                return [max_row, mid_col]
            elif matrix[max_row][mid_col] < left_val:
                right = mid_col - 1
            else:
                left = mid_col + 1
        
        return [-1, -1]

# Test Examples
def run_examples():
    mst = MatrixSearchingTechniques()
    
    print("=== MATRIX SEARCHING TECHNIQUES ===\n")
    
    matrix = [
        [1,  4,  7,  11],
        [2,  5,  8,  12],
        [3,  6,  9,  16],
        [10, 13, 14, 17]
    ]
    
    print("Matrix:")
    for row in matrix:
        print(row)
    
    target = 5
    result = mst.search_row_col_sorted(matrix, target)
    print(f"\nSearch for {target}: {result}")

if __name__ == "__main__":
    run_examples() 