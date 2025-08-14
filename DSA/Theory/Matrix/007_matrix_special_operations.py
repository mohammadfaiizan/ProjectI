"""
Matrix Special Operations
=========================

Topics: Matrix multiplication optimizations, special matrix operations
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List, Tuple

class MatrixSpecialOperations:
    
    # ==========================================
    # 1. SET MATRIX ZEROS
    # ==========================================
    
    def set_zeros(self, matrix: List[List[int]]) -> None:
        """LC 73: Set Matrix Zeroes - in-place solution"""
        if not matrix or not matrix[0]:
            return
        
        m, n = len(matrix), len(matrix[0])
        first_row_zero = any(matrix[0][j] == 0 for j in range(n))
        first_col_zero = any(matrix[i][0] == 0 for i in range(m))
        
        # Use first row and column as markers
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        # Set zeros based on markers
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        # Handle first row and column
        if first_row_zero:
            for j in range(n):
                matrix[0][j] = 0
        
        if first_col_zero:
            for i in range(m):
                matrix[i][0] = 0
    
    # ==========================================
    # 2. MATRIX DETERMINANT
    # ==========================================
    
    def matrix_determinant(self, matrix: List[List[int]]) -> int:
        """Calculate determinant using cofactor expansion"""
        n = len(matrix)
        
        if n == 1:
            return matrix[0][0]
        
        if n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        
        det = 0
        for j in range(n):
            # Create minor matrix
            minor = []
            for i in range(1, n):
                row = []
                for k in range(n):
                    if k != j:
                        row.append(matrix[i][k])
                minor.append(row)
            
            # Calculate cofactor
            cofactor = ((-1) ** j) * matrix[0][j] * self.matrix_determinant(minor)
            det += cofactor
        
        return det
    
    # ==========================================
    # 3. MATRIX POWER
    # ==========================================
    
    def matrix_power(self, matrix: List[List[int]], n: int) -> List[List[int]]:
        """Calculate matrix^n using fast exponentiation"""
        size = len(matrix)
        
        def matrix_multiply(A, B):
            result = [[0] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        result[i][j] += A[i][k] * B[k][j]
            return result
        
        def matrix_identity():
            identity = [[0] * size for _ in range(size)]
            for i in range(size):
                identity[i][i] = 1
            return identity
        
        if n == 0:
            return matrix_identity()
        
        if n == 1:
            return matrix
        
        if n % 2 == 0:
            half_power = self.matrix_power(matrix, n // 2)
            return matrix_multiply(half_power, half_power)
        else:
            return matrix_multiply(matrix, self.matrix_power(matrix, n - 1))
    
    # ==========================================
    # 4. TOEPLITZ MATRIX
    # ==========================================
    
    def is_toeplitz_matrix(self, matrix: List[List[int]]) -> bool:
        """LC 766: Toeplitz Matrix"""
        m, n = len(matrix), len(matrix[0])
        
        for i in range(m - 1):
            for j in range(n - 1):
                if matrix[i][j] != matrix[i + 1][j + 1]:
                    return False
        
        return True
    
    def toeplitz_matrix_follow_up(self, matrix: List[List[int]]) -> bool:
        """Toeplitz matrix check with memory constraints"""
        if not matrix or not matrix[0]:
            return True
        
        # Only keep previous row in memory
        prev_row = matrix[0]
        
        for i in range(1, len(matrix)):
            curr_row = matrix[i]
            
            # Check diagonal elements
            for j in range(len(curr_row) - 1):
                if j + 1 < len(prev_row) and curr_row[j] != prev_row[j + 1]:
                    return False
            
            prev_row = curr_row
        
        return True
    
    # ==========================================
    # 5. SCALAR MATRIX OPERATIONS
    # ==========================================
    
    def scalar_matrix_multiply(self, matrix: List[List[int]], scalar: int) -> List[List[int]]:
        """Multiply matrix by scalar"""
        m, n = len(matrix), len(matrix[0])
        result = [[0] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                result[i][j] = matrix[i][j] * scalar
        
        return result
    
    def matrix_sum(self, matrices: List[List[List[int]]]) -> List[List[int]]:
        """Sum multiple matrices"""
        if not matrices:
            return []
        
        m, n = len(matrices[0]), len(matrices[0][0])
        result = [[0] * n for _ in range(m)]
        
        for matrix in matrices:
            for i in range(m):
                for j in range(n):
                    result[i][j] += matrix[i][j]
        
        return result
    
    # ==========================================
    # 6. SPECIAL MATRIX TYPES
    # ==========================================
    
    def is_magic_square(self, matrix: List[List[int]]) -> bool:
        """Check if matrix is a magic square"""
        n = len(matrix)
        
        if n != len(matrix[0]):
            return False
        
        # Calculate target sum (first row sum)
        target_sum = sum(matrix[0])
        
        # Check all rows
        for i in range(n):
            if sum(matrix[i]) != target_sum:
                return False
        
        # Check all columns
        for j in range(n):
            if sum(matrix[i][j] for i in range(n)) != target_sum:
                return False
        
        # Check main diagonal
        if sum(matrix[i][i] for i in range(n)) != target_sum:
            return False
        
        # Check anti-diagonal
        if sum(matrix[i][n - 1 - i] for i in range(n)) != target_sum:
            return False
        
        return True
    
    def generate_magic_square_3x3(self) -> List[List[int]]:
        """Generate 3x3 magic square"""
        return [
            [2, 7, 6],
            [9, 5, 1],
            [4, 3, 8]
        ]
    
    def is_symmetric(self, matrix: List[List[int]]) -> bool:
        """Check if matrix is symmetric"""
        n = len(matrix)
        
        if n != len(matrix[0]):
            return False
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != matrix[j][i]:
                    return False
        
        return True
    
    def is_antisymmetric(self, matrix: List[List[int]]) -> bool:
        """Check if matrix is antisymmetric"""
        n = len(matrix)
        
        if n != len(matrix[0]):
            return False
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != -matrix[j][i]:
                    return False
        
        return True

# Test Examples
def run_examples():
    mso = MatrixSpecialOperations()
    
    print("=== MATRIX SPECIAL OPERATIONS ===\n")
    
    # Set Matrix Zeros
    print("1. SET MATRIX ZEROS:")
    matrix = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    
    print("Original matrix:")
    for row in matrix:
        print(row)
    
    mso.set_zeros(matrix)
    print("\nAfter setting zeros:")
    for row in matrix:
        print(row)
    
    # Matrix Determinant
    print("\n2. MATRIX DETERMINANT:")
    det_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    det = mso.matrix_determinant(det_matrix)
    print(f"Determinant: {det}")
    
    # Toeplitz Matrix
    print("\n3. TOEPLITZ MATRIX:")
    toeplitz = [
        [1, 2, 3, 4],
        [5, 1, 2, 3],
        [9, 5, 1, 2]
    ]
    
    is_toeplitz = mso.is_toeplitz_matrix(toeplitz)
    print(f"Is Toeplitz: {is_toeplitz}")
    
    # Magic Square
    print("\n4. MAGIC SQUARE:")
    magic = mso.generate_magic_square_3x3()
    print("Generated 3x3 magic square:")
    for row in magic:
        print(row)
    
    is_magic = mso.is_magic_square(magic)
    print(f"Is magic square: {is_magic}")

if __name__ == "__main__":
    run_examples() 