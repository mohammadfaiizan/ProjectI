"""
Array Matrix Operations - 2D Array Manipulation
===============================================

Topics: Matrix traversal, rotation, transpose, spiral order
Companies: Google, Facebook, Amazon, Microsoft, Apple
Difficulty: Medium to Hard
"""

from typing import List, Tuple

class ArrayMatrixOperations:
    
    # ==========================================
    # 1. BASIC MATRIX OPERATIONS
    # ==========================================
    
    def matrix_transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        """Transpose matrix - swap rows and columns
        Time: O(m*n), Space: O(m*n)
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        result = [[0] * m for _ in range(n)]
        
        for i in range(m):
            for j in range(n):
                result[j][i] = matrix[i][j]
        
        return result
    
    def rotate_matrix_90_clockwise(self, matrix: List[List[int]]) -> None:
        """LC 48: Rotate Image - 90 degrees clockwise in-place
        Time: O(n²), Space: O(1)
        """
        n = len(matrix)
        
        # Transpose matrix
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # Reverse each row
        for i in range(n):
            matrix[i].reverse()
    
    def rotate_matrix_90_counterclockwise(self, matrix: List[List[int]]) -> None:
        """Rotate matrix 90 degrees counterclockwise in-place
        Time: O(n²), Space: O(1)
        """
        n = len(matrix)
        
        # Reverse each row first
        for i in range(n):
            matrix[i].reverse()
        
        # Transpose matrix
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    def spiral_order(self, matrix: List[List[int]]) -> List[int]:
        """LC 54: Spiral Matrix - traverse in spiral order
        Time: O(m*n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return []
        
        result = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        
        while top <= bottom and left <= right:
            # Traverse right
            for col in range(left, right + 1):
                result.append(matrix[top][col])
            top += 1
            
            # Traverse down
            for row in range(top, bottom + 1):
                result.append(matrix[row][right])
            right -= 1
            
            # Traverse left (if still valid)
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    result.append(matrix[bottom][col])
                bottom -= 1
            
            # Traverse up (if still valid)
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    result.append(matrix[row][left])
                left += 1
        
        return result
    
    def generate_spiral_matrix(self, n: int) -> List[List[int]]:
        """LC 59: Spiral Matrix II - generate spiral matrix
        Time: O(n²), Space: O(n²)
        """
        matrix = [[0] * n for _ in range(n)]
        num = 1
        top, bottom = 0, n - 1
        left, right = 0, n - 1
        
        while top <= bottom and left <= right:
            # Fill right
            for col in range(left, right + 1):
                matrix[top][col] = num
                num += 1
            top += 1
            
            # Fill down
            for row in range(top, bottom + 1):
                matrix[row][right] = num
                num += 1
            right -= 1
            
            # Fill left
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    matrix[bottom][col] = num
                    num += 1
                bottom -= 1
            
            # Fill up
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    matrix[row][left] = num
                    num += 1
                left += 1
        
        return matrix
    
    # ==========================================
    # 2. MATRIX SEARCH OPERATIONS
    # ==========================================
    
    def search_2d_matrix_i(self, matrix: List[List[int]], target: int) -> bool:
        """LC 74: Search 2D Matrix (row-wise and column-wise sorted)
        Time: O(log(m*n)), Space: O(1)
        """
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
    
    def search_2d_matrix_ii(self, matrix: List[List[int]], target: int) -> bool:
        """LC 240: Search 2D Matrix II (sorted rows and columns)
        Time: O(m + n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return False
        
        row, col = 0, len(matrix[0]) - 1
        
        while row < len(matrix) and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False
    
    # ==========================================
    # 3. MATRIX MODIFICATION PROBLEMS
    # ==========================================
    
    def set_matrix_zeros(self, matrix: List[List[int]]) -> None:
        """LC 73: Set Matrix Zeroes - in-place
        Time: O(m*n), Space: O(1)
        """
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
        
        # Handle first row
        if first_row_zero:
            for j in range(n):
                matrix[0][j] = 0
        
        # Handle first column
        if first_col_zero:
            for i in range(m):
                matrix[i][0] = 0
    
    def diagonal_traverse(self, matrix: List[List[int]]) -> List[int]:
        """LC 498: Diagonal Traverse
        Time: O(m*n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        result = []
        
        for d in range(m + n - 1):
            intermediate = []
            
            # Determine start point of diagonal
            if d < n:
                row, col = 0, d
            else:
                row, col = d - n + 1, n - 1
            
            # Traverse diagonal
            while row < m and col >= 0:
                intermediate.append(matrix[row][col])
                row += 1
                col -= 1
            
            # Reverse for upward diagonals
            if d % 2 == 0:
                intermediate.reverse()
            
            result.extend(intermediate)
        
        return result
    
    # ==========================================
    # 4. MATRIX SPECIAL PROBLEMS
    # ==========================================
    
    def lucky_numbers(self, matrix: List[List[int]]) -> List[int]:
        """LC 1380: Lucky Numbers in Matrix
        Time: O(m*n), Space: O(m + n)
        """
        m, n = len(matrix), len(matrix[0])
        
        # Find min in each row
        row_mins = [min(row) for row in matrix]
        
        # Find max in each column
        col_maxs = [max(matrix[i][j] for i in range(m)) for j in range(n)]
        
        result = []
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == row_mins[i] and matrix[i][j] == col_maxs[j]:
                    result.append(matrix[i][j])
        
        return result
    
    def toeplitz_matrix(self, matrix: List[List[int]]) -> bool:
        """LC 766: Toeplitz Matrix - all diagonals have same elements
        Time: O(m*n), Space: O(1)
        """
        m, n = len(matrix), len(matrix[0])
        
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] != matrix[i-1][j-1]:
                    return False
        
        return True
    
    def largest_rectangle_binary_matrix(self, matrix: List[List[str]]) -> int:
        """LC 85: Maximal Rectangle in binary matrix
        Time: O(m*n), Space: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        def largest_rectangle_histogram(heights):
            stack = []
            max_area = 0
            
            for i, h in enumerate(heights):
                while stack and heights[stack[-1]] > h:
                    height = heights[stack.pop()]
                    width = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, height * width)
                stack.append(i)
            
            while stack:
                height = heights[stack.pop()]
                width = len(heights) if not stack else len(heights) - stack[-1] - 1
                max_area = max(max_area, height * width)
            
            return max_area
        
        max_area = 0
        heights = [0] * len(matrix[0])
        
        for row in matrix:
            for j, val in enumerate(row):
                heights[j] = heights[j] + 1 if val == '1' else 0
            
            max_area = max(max_area, largest_rectangle_histogram(heights))
        
        return max_area
    
    def count_square_submatrices(self, matrix: List[List[int]]) -> int:
        """LC 1277: Count Square Submatrices with All Ones
        Time: O(m*n), Space: O(1)
        """
        if not matrix:
            return 0
        
        count = 0
        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 1:
                    if i > 0 and j > 0:
                        matrix[i][j] = min(matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]) + 1
                    
                    count += matrix[i][j]
        
        return count
    
    # ==========================================
    # 5. GAME OF LIFE & SIMULATION
    # ==========================================
    
    def game_of_life(self, board: List[List[int]]) -> None:
        """LC 289: Game of Life - in-place simulation
        Time: O(m*n), Space: O(1)
        """
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        
        # Use additional bits to store next state
        # 00: dead -> dead, 01: dead -> live, 10: live -> dead, 11: live -> live
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i in range(m):
            for j in range(n):
                live_neighbors = 0
                
                # Count live neighbors
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n and board[ni][nj] & 1:
                        live_neighbors += 1
                
                # Apply rules
                if board[i][j] & 1:  # Currently alive
                    if live_neighbors in [2, 3]:
                        board[i][j] |= 2  # Will stay alive
                else:  # Currently dead
                    if live_neighbors == 3:
                        board[i][j] |= 2  # Will become alive
        
        # Update to next state
        for i in range(m):
            for j in range(n):
                board[i][j] >>= 1

# Test Examples
def run_examples():
    amo = ArrayMatrixOperations()
    
    print("=== ARRAY MATRIX OPERATIONS EXAMPLES ===\n")
    
    # Matrix rotation
    print("1. MATRIX ROTATION:")
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f"Original matrix: {matrix}")
    
    matrix_copy = [row[:] for row in matrix]
    amo.rotate_matrix_90_clockwise(matrix_copy)
    print(f"90° clockwise: {matrix_copy}")
    
    # Spiral traversal
    print("\n2. SPIRAL TRAVERSAL:")
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    spiral = amo.spiral_order(matrix)
    print(f"Matrix: {matrix}")
    print(f"Spiral order: {spiral}")
    
    # Generate spiral matrix
    print("\n3. GENERATE SPIRAL MATRIX:")
    spiral_matrix = amo.generate_spiral_matrix(3)
    print(f"3x3 spiral matrix: {spiral_matrix}")
    
    # Matrix search
    print("\n4. MATRIX SEARCH:")
    matrix = [[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16]]
    target = 5
    found = amo.search_2d_matrix_ii(matrix, target)
    print(f"Search {target} in matrix: {found}")
    
    # Set zeros
    print("\n5. SET MATRIX ZEROS:")
    matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    print(f"Before: {matrix}")
    amo.set_matrix_zeros(matrix)
    print(f"After: {matrix}")
    
    # Diagonal traverse
    print("\n6. DIAGONAL TRAVERSE:")
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    diagonal = amo.diagonal_traverse(matrix)
    print(f"Diagonal traversal: {diagonal}")

if __name__ == "__main__":
    run_examples() 