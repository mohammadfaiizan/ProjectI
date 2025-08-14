"""
Matrix Transformations
======================

Topics: Rotation, reflection, transpose operations
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium
"""

from typing import List

class MatrixTransformations:
    
    # ==========================================
    # 1. ROTATION OPERATIONS
    # ==========================================
    
    def rotate_90_clockwise(self, matrix: List[List[int]]) -> None:
        """LC 48: Rotate matrix 90 degrees clockwise in-place"""
        n = len(matrix)
        
        # Transpose the matrix
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # Reverse each row
        for i in range(n):
            matrix[i].reverse()
    
    def rotate_90_counterclockwise(self, matrix: List[List[int]]) -> None:
        """Rotate matrix 90 degrees counterclockwise in-place"""
        n = len(matrix)
        
        # Reverse each row first
        for i in range(n):
            matrix[i].reverse()
        
        # Transpose the matrix
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    def rotate_180(self, matrix: List[List[int]]) -> None:
        """Rotate matrix 180 degrees in-place"""
        m, n = len(matrix), len(matrix[0])
        
        for i in range(m):
            for j in range(n // 2):
                matrix[i][j], matrix[i][n - 1 - j] = matrix[i][n - 1 - j], matrix[i][j]
        
        for i in range(m // 2):
            matrix[i], matrix[m - 1 - i] = matrix[m - 1 - i], matrix[i]
    
    # ==========================================
    # 2. REFLECTION OPERATIONS
    # ==========================================
    
    def reflect_horizontal(self, matrix: List[List[int]]) -> None:
        """Reflect matrix horizontally (flip left-right)"""
        for row in matrix:
            row.reverse()
    
    def reflect_vertical(self, matrix: List[List[int]]) -> None:
        """Reflect matrix vertically (flip up-down)"""
        matrix.reverse()
    
    def reflect_main_diagonal(self, matrix: List[List[int]]) -> None:
        """Reflect across main diagonal (transpose)"""
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    def reflect_anti_diagonal(self, matrix: List[List[int]]) -> None:
        """Reflect across anti-diagonal"""
        n = len(matrix)
        
        # First reverse rows
        matrix.reverse()
        
        # Then transpose
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # Reverse rows again
        matrix.reverse()
    
    # ==========================================
    # 3. ADVANCED TRANSFORMATIONS
    # ==========================================
    
    def spiral_to_linear(self, matrix: List[List[int]]) -> List[int]:
        """Convert matrix to 1D array in spiral order"""
        if not matrix or not matrix[0]:
            return []
        
        result = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        
        while top <= bottom and left <= right:
            for col in range(left, right + 1):
                result.append(matrix[top][col])
            top += 1
            
            for row in range(top, bottom + 1):
                result.append(matrix[row][right])
            right -= 1
            
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    result.append(matrix[bottom][col])
                bottom -= 1
            
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    result.append(matrix[row][left])
                left += 1
        
        return result
    
    def reshape_matrix(self, matrix: List[List[int]], r: int, c: int) -> List[List[int]]:
        """LC 566: Reshape the Matrix"""
        m, n = len(matrix), len(matrix[0])
        
        if m * n != r * c:
            return matrix
        
        flat = []
        for row in matrix:
            flat.extend(row)
        
        result = []
        for i in range(r):
            result.append(flat[i * c:(i + 1) * c])
        
        return result

# Test Examples
def run_examples():
    mt = MatrixTransformations()
    
    print("=== MATRIX TRANSFORMATIONS ===\n")
    
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    print("Original Matrix:")
    for row in matrix:
        print(row)
    
    # Test rotation
    matrix_copy = [row[:] for row in matrix]
    mt.rotate_90_clockwise(matrix_copy)
    print("\nAfter 90° Clockwise Rotation:")
    for row in matrix_copy:
        print(row)

if __name__ == "__main__":
    run_examples() 