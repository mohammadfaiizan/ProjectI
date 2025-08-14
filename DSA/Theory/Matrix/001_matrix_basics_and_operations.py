"""
Matrix Basics and Operations
===========================

Topics: Matrix creation, basic operations, element access, manipulation
Companies: Google, Facebook, Amazon, Microsoft, Apple
Difficulty: Easy to Medium
"""

from typing import List, Tuple, Optional
import random

class MatrixBasicsOperations:
    
    # ==========================================
    # 1. MATRIX CREATION AND INITIALIZATION
    # ==========================================
    
    def create_matrix(self, rows: int, cols: int, default_value: int = 0) -> List[List[int]]:
        """Create matrix with given dimensions and default value
        Time: O(m*n), Space: O(m*n)
        """
        return [[default_value for _ in range(cols)] for _ in range(rows)]
    
    def create_identity_matrix(self, n: int) -> List[List[int]]:
        """Create n×n identity matrix
        Time: O(n²), Space: O(n²)
        """
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1
        return matrix
    
    def create_random_matrix(self, rows: int, cols: int, min_val: int = 0, max_val: int = 100) -> List[List[int]]:
        """Create matrix with random values
        Time: O(m*n), Space: O(m*n)
        """
        return [[random.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]
    
    def matrix_from_list(self, data: List[int], rows: int, cols: int) -> List[List[int]]:
        """Convert 1D list to 2D matrix
        Time: O(m*n), Space: O(m*n)
        """
        if len(data) != rows * cols:
            raise ValueError("Data length doesn't match matrix dimensions")
        
        matrix = []
        for i in range(rows):
            row = data[i * cols:(i + 1) * cols]
            matrix.append(row)
        return matrix
    
    # ==========================================
    # 2. BASIC MATRIX OPERATIONS
    # ==========================================
    
    def matrix_addition(self, matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
        """Add two matrices
        Time: O(m*n), Space: O(m*n)
        """
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Matrices must have same dimensions")
        
        rows, cols = len(matrix1), len(matrix1[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = matrix1[i][j] + matrix2[i][j]
        
        return result
    
    def matrix_subtraction(self, matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
        """Subtract two matrices
        Time: O(m*n), Space: O(m*n)
        """
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Matrices must have same dimensions")
        
        rows, cols = len(matrix1), len(matrix1[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = matrix1[i][j] - matrix2[i][j]
        
        return result
    
    def matrix_multiplication(self, matrix1: List[List[int]], matrix2: List[List[int]]) -> List[List[int]]:
        """Multiply two matrices
        Time: O(m*n*p), Space: O(m*p)
        """
        rows1, cols1 = len(matrix1), len(matrix1[0])
        rows2, cols2 = len(matrix2), len(matrix2[0])
        
        if cols1 != rows2:
            raise ValueError("Number of columns in first matrix must equal rows in second")
        
        result = [[0 for _ in range(cols2)] for _ in range(rows1)]
        
        for i in range(rows1):
            for j in range(cols2):
                for k in range(cols1):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        
        return result
    
    def scalar_multiplication(self, matrix: List[List[int]], scalar: int) -> List[List[int]]:
        """Multiply matrix by scalar
        Time: O(m*n), Space: O(m*n)
        """
        rows, cols = len(matrix), len(matrix[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = matrix[i][j] * scalar
        
        return result
    
    # ==========================================
    # 3. MATRIX PROPERTIES AND ANALYSIS
    # ==========================================
    
    def is_square_matrix(self, matrix: List[List[int]]) -> bool:
        """Check if matrix is square
        Time: O(1), Space: O(1)
        """
        return len(matrix) == len(matrix[0])
    
    def is_symmetric_matrix(self, matrix: List[List[int]]) -> bool:
        """Check if matrix is symmetric
        Time: O(n²), Space: O(1)
        """
        if not self.is_square_matrix(matrix):
            return False
        
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != matrix[j][i]:
                    return False
        return True
    
    def is_diagonal_matrix(self, matrix: List[List[int]]) -> bool:
        """Check if matrix is diagonal
        Time: O(n²), Space: O(1)
        """
        if not self.is_square_matrix(matrix):
            return False
        
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i][j] != 0:
                    return False
        return True
    
    def matrix_trace(self, matrix: List[List[int]]) -> int:
        """Calculate trace (sum of diagonal elements)
        Time: O(n), Space: O(1)
        """
        if not self.is_square_matrix(matrix):
            raise ValueError("Matrix must be square")
        
        trace = 0
        for i in range(len(matrix)):
            trace += matrix[i][i]
        return trace
    
    def matrix_determinant_2x2(self, matrix: List[List[int]]) -> int:
        """Calculate determinant for 2x2 matrix
        Time: O(1), Space: O(1)
        """
        if len(matrix) != 2 or len(matrix[0]) != 2:
            raise ValueError("Matrix must be 2x2")
        
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # ==========================================
    # 4. MATRIX TRAVERSAL BASICS
    # ==========================================
    
    def row_wise_traversal(self, matrix: List[List[int]]) -> List[int]:
        """Traverse matrix row by row
        Time: O(m*n), Space: O(m*n)
        """
        result = []
        for row in matrix:
            result.extend(row)
        return result
    
    def column_wise_traversal(self, matrix: List[List[int]]) -> List[int]:
        """Traverse matrix column by column
        Time: O(m*n), Space: O(m*n)
        """
        if not matrix or not matrix[0]:
            return []
        
        result = []
        rows, cols = len(matrix), len(matrix[0])
        
        for j in range(cols):
            for i in range(rows):
                result.append(matrix[i][j])
        
        return result
    
    def diagonal_traversal(self, matrix: List[List[int]]) -> Tuple[List[int], List[int]]:
        """Traverse main and anti-diagonal
        Time: O(min(m,n)), Space: O(min(m,n))
        """
        if not matrix or not matrix[0]:
            return [], []
        
        rows, cols = len(matrix), len(matrix[0])
        main_diagonal = []
        anti_diagonal = []
        
        # Main diagonal (top-left to bottom-right)
        for i in range(min(rows, cols)):
            main_diagonal.append(matrix[i][i])
        
        # Anti-diagonal (top-right to bottom-left)
        for i in range(min(rows, cols)):
            anti_diagonal.append(matrix[i][cols - 1 - i])
        
        return main_diagonal, anti_diagonal
    
    # ==========================================
    # 5. MATRIX SEARCHING BASICS
    # ==========================================
    
    def linear_search(self, matrix: List[List[int]], target: int) -> Optional[Tuple[int, int]]:
        """Linear search in matrix
        Time: O(m*n), Space: O(1)
        """
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == target:
                    return (i, j)
        return None
    
    def find_min_max(self, matrix: List[List[int]]) -> Tuple[int, int]:
        """Find minimum and maximum elements
        Time: O(m*n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            raise ValueError("Matrix is empty")
        
        min_val = max_val = matrix[0][0]
        
        for row in matrix:
            for val in row:
                min_val = min(min_val, val)
                max_val = max(max_val, val)
        
        return min_val, max_val
    
    def count_elements(self, matrix: List[List[int]], target: int) -> int:
        """Count occurrences of target element
        Time: O(m*n), Space: O(1)
        """
        count = 0
        for row in matrix:
            for val in row:
                if val == target:
                    count += 1
        return count
    
    # ==========================================
    # 6. MATRIX MANIPULATION BASICS
    # ==========================================
    
    def transpose_matrix(self, matrix: List[List[int]]) -> List[List[int]]:
        """Transpose matrix (swap rows and columns)
        Time: O(m*n), Space: O(m*n)
        """
        if not matrix or not matrix[0]:
            return []
        
        rows, cols = len(matrix), len(matrix[0])
        transposed = [[0 for _ in range(rows)] for _ in range(cols)]
        
        for i in range(rows):
            for j in range(cols):
                transposed[j][i] = matrix[i][j]
        
        return transposed
    
    def reverse_rows(self, matrix: List[List[int]]) -> List[List[int]]:
        """Reverse each row of matrix
        Time: O(m*n), Space: O(1) in-place
        """
        for row in matrix:
            row.reverse()
        return matrix
    
    def reverse_columns(self, matrix: List[List[int]]) -> List[List[int]]:
        """Reverse columns (flip vertically)
        Time: O(m*n), Space: O(1) in-place
        """
        matrix.reverse()
        return matrix
    
    def flatten_matrix(self, matrix: List[List[int]]) -> List[int]:
        """Convert 2D matrix to 1D list
        Time: O(m*n), Space: O(m*n)
        """
        result = []
        for row in matrix:
            result.extend(row)
        return result
    
    # ==========================================
    # 7. BOUNDARY AND EDGE OPERATIONS
    # ==========================================
    
    def get_boundary_elements(self, matrix: List[List[int]]) -> List[int]:
        """Get all boundary elements
        Time: O(m+n), Space: O(m+n)
        """
        if not matrix or not matrix[0]:
            return []
        
        rows, cols = len(matrix), len(matrix[0])
        boundary = []
        
        if rows == 1:
            return matrix[0]
        if cols == 1:
            return [row[0] for row in matrix]
        
        # Top row
        boundary.extend(matrix[0])
        
        # Right column (excluding corners)
        for i in range(1, rows - 1):
            boundary.append(matrix[i][cols - 1])
        
        # Bottom row (in reverse order)
        if rows > 1:
            boundary.extend(matrix[rows - 1][::-1])
        
        # Left column (excluding corners, from bottom to top)
        for i in range(rows - 2, 0, -1):
            boundary.append(matrix[i][0])
        
        return boundary
    
    def set_boundary_zeros(self, matrix: List[List[int]]) -> List[List[int]]:
        """Set all boundary elements to zero
        Time: O(m*n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return matrix
        
        rows, cols = len(matrix), len(matrix[0])
        
        # Set first and last row to zero
        for j in range(cols):
            matrix[0][j] = 0
            matrix[rows - 1][j] = 0
        
        # Set first and last column to zero
        for i in range(rows):
            matrix[i][0] = 0
            matrix[i][cols - 1] = 0
        
        return matrix

# Test Examples
def run_examples():
    mbo = MatrixBasicsOperations()
    
    print("=== MATRIX BASICS AND OPERATIONS EXAMPLES ===\n")
    
    # Matrix creation
    print("1. MATRIX CREATION:")
    matrix3x3 = mbo.create_matrix(3, 3, 5)
    print(f"3x3 matrix with default value 5:")
    for row in matrix3x3:
        print(row)
    
    identity = mbo.create_identity_matrix(3)
    print(f"\n3x3 Identity matrix:")
    for row in identity:
        print(row)
    
    # Basic operations
    print("\n2. BASIC OPERATIONS:")
    matrix1 = [[1, 2], [3, 4]]
    matrix2 = [[5, 6], [7, 8]]
    
    addition = mbo.matrix_addition(matrix1, matrix2)
    print(f"Matrix addition: {addition}")
    
    multiplication = mbo.matrix_multiplication(matrix1, matrix2)
    print(f"Matrix multiplication: {multiplication}")
    
    # Matrix properties
    print("\n3. MATRIX PROPERTIES:")
    symmetric = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
    print(f"Is symmetric: {mbo.is_symmetric_matrix(symmetric)}")
    
    trace = mbo.matrix_trace(identity)
    print(f"Trace of identity matrix: {trace}")
    
    # Traversal
    print("\n4. MATRIX TRAVERSAL:")
    test_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f"Original matrix:")
    for row in test_matrix:
        print(row)
    
    row_wise = mbo.row_wise_traversal(test_matrix)
    print(f"Row-wise traversal: {row_wise}")
    
    col_wise = mbo.column_wise_traversal(test_matrix)
    print(f"Column-wise traversal: {col_wise}")
    
    main_diag, anti_diag = mbo.diagonal_traversal(test_matrix)
    print(f"Main diagonal: {main_diag}")
    print(f"Anti-diagonal: {anti_diag}")
    
    # Searching
    print("\n5. MATRIX SEARCHING:")
    position = mbo.linear_search(test_matrix, 5)
    print(f"Position of 5: {position}")
    
    min_val, max_val = mbo.find_min_max(test_matrix)
    print(f"Min: {min_val}, Max: {max_val}")
    
    # Manipulation
    print("\n6. MATRIX MANIPULATION:")
    transposed = mbo.transpose_matrix(test_matrix)
    print(f"Transposed matrix:")
    for row in transposed:
        print(row)
    
    boundary = mbo.get_boundary_elements(test_matrix)
    print(f"Boundary elements: {boundary}")

if __name__ == "__main__":
    run_examples() 