"""
Matrix Traversal Patterns
=========================

Topics: Spiral, zigzag, diagonal, wave traversals
Companies: Google, Facebook, Amazon, Microsoft, Apple
Difficulty: Medium
"""

from typing import List, Generator

class MatrixTraversalPatterns:
    
    # ==========================================
    # 1. SPIRAL TRAVERSAL PATTERNS
    # ==========================================
    
    def spiral_traversal_clockwise(self, matrix: List[List[int]]) -> List[int]:
        """LC 54: Spiral Matrix - Clockwise traversal
        Time: O(m*n), Space: O(1) excluding output
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
            
            # Traverse left (if we still have rows)
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    result.append(matrix[bottom][col])
                bottom -= 1
            
            # Traverse up (if we still have columns)
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    result.append(matrix[row][left])
                left += 1
        
        return result
    
    def spiral_traversal_counterclockwise(self, matrix: List[List[int]]) -> List[int]:
        """Spiral traversal in counter-clockwise direction
        Time: O(m*n), Space: O(1) excluding output
        """
        if not matrix or not matrix[0]:
            return []
        
        result = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        
        while top <= bottom and left <= right:
            # Traverse down
            for row in range(top, bottom + 1):
                result.append(matrix[row][left])
            left += 1
            
            # Traverse right
            for col in range(left, right + 1):
                result.append(matrix[bottom][col])
            bottom -= 1
            
            # Traverse up (if we still have columns)
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    result.append(matrix[row][right])
                right -= 1
            
            # Traverse left (if we still have rows)
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    result.append(matrix[top][col])
                top += 1
        
        return result
    
    def generate_spiral_matrix(self, n: int) -> List[List[int]]:
        """LC 59: Spiral Matrix II - Generate spiral matrix
        Time: O(n²), Space: O(n²)
        """
        matrix = [[0] * n for _ in range(n)]
        top, bottom = 0, n - 1
        left, right = 0, n - 1
        num = 1
        
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
    # 2. DIAGONAL TRAVERSAL PATTERNS
    # ==========================================
    
    def diagonal_traversal_top_to_bottom(self, matrix: List[List[int]]) -> List[int]:
        """LC 498: Diagonal Traverse - Top-left to bottom-right diagonals
        Time: O(m*n), Space: O(1) excluding output
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        result = []
        going_up = True
        
        for d in range(m + n - 1):
            if going_up:
                # Going up: start from bottom of diagonal
                row = min(d, m - 1)
                col = d - row
                while row >= 0 and col < n:
                    result.append(matrix[row][col])
                    row -= 1
                    col += 1
            else:
                # Going down: start from top of diagonal
                col = min(d, n - 1)
                row = d - col
                while col >= 0 and row < m:
                    result.append(matrix[row][col])
                    row += 1
                    col -= 1
            
            going_up = not going_up
        
        return result
    
    def anti_diagonal_traversal(self, matrix: List[List[int]]) -> List[List[int]]:
        """Traverse all anti-diagonals (top-right to bottom-left)
        Time: O(m*n), Space: O(m*n)
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        diagonals = []
        
        # Upper half (including main anti-diagonal)
        for col in range(n):
            diagonal = []
            r, c = 0, col
            while r < m and c >= 0:
                diagonal.append(matrix[r][c])
                r += 1
                c -= 1
            diagonals.append(diagonal)
        
        # Lower half
        for row in range(1, m):
            diagonal = []
            r, c = row, n - 1
            while r < m and c >= 0:
                diagonal.append(matrix[r][c])
                r += 1
                c -= 1
            diagonals.append(diagonal)
        
        return diagonals
    
    def main_diagonal_traversal(self, matrix: List[List[int]]) -> List[List[int]]:
        """Traverse all main diagonals (top-left to bottom-right)
        Time: O(m*n), Space: O(m*n)
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        diagonals = []
        
        # Upper half (including main diagonal)
        for col in range(n):
            diagonal = []
            r, c = 0, col
            while r < m and c < n:
                diagonal.append(matrix[r][c])
                r += 1
                c += 1
            diagonals.append(diagonal)
        
        # Lower half
        for row in range(1, m):
            diagonal = []
            r, c = row, 0
            while r < m and c < n:
                diagonal.append(matrix[r][c])
                r += 1
                c += 1
            diagonals.append(diagonal)
        
        return diagonals
    
    # ==========================================
    # 3. ZIGZAG/WAVE TRAVERSAL PATTERNS
    # ==========================================
    
    def zigzag_row_traversal(self, matrix: List[List[int]]) -> List[int]:
        """Zigzag traversal by rows (alternating left-right, right-left)
        Time: O(m*n), Space: O(1) excluding output
        """
        if not matrix or not matrix[0]:
            return []
        
        result = []
        
        for i, row in enumerate(matrix):
            if i % 2 == 0:
                # Even rows: left to right
                result.extend(row)
            else:
                # Odd rows: right to left
                result.extend(row[::-1])
        
        return result
    
    def zigzag_column_traversal(self, matrix: List[List[int]]) -> List[int]:
        """Zigzag traversal by columns
        Time: O(m*n), Space: O(1) excluding output
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        result = []
        
        for j in range(n):
            if j % 2 == 0:
                # Even columns: top to bottom
                for i in range(m):
                    result.append(matrix[i][j])
            else:
                # Odd columns: bottom to top
                for i in range(m - 1, -1, -1):
                    result.append(matrix[i][j])
        
        return result
    
    def wave_traversal(self, matrix: List[List[int]]) -> List[int]:
        """Wave traversal (snake pattern)
        Time: O(m*n), Space: O(1) excluding output
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        result = []
        
        for i in range(m):
            if i % 2 == 0:
                # Even rows: left to right
                for j in range(n):
                    result.append(matrix[i][j])
            else:
                # Odd rows: right to left
                for j in range(n - 1, -1, -1):
                    result.append(matrix[i][j])
        
        return result
    
    # ==========================================
    # 4. LAYER-BASED TRAVERSAL
    # ==========================================
    
    def layer_wise_traversal(self, matrix: List[List[int]]) -> List[List[int]]:
        """Traverse matrix layer by layer (concentric rectangles)
        Time: O(m*n), Space: O(m*n)
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        layers = []
        
        top, bottom = 0, m - 1
        left, right = 0, n - 1
        
        while top <= bottom and left <= right:
            layer = []
            
            if top == bottom:
                # Single row
                layer.extend(matrix[top][left:right + 1])
            elif left == right:
                # Single column
                for i in range(top, bottom + 1):
                    layer.append(matrix[i][left])
            else:
                # Full rectangle
                # Top row
                layer.extend(matrix[top][left:right + 1])
                
                # Right column (excluding top corner)
                for i in range(top + 1, bottom + 1):
                    layer.append(matrix[i][right])
                
                # Bottom row (excluding right corner, in reverse)
                if bottom > top:
                    layer.extend(matrix[bottom][left:right][::-1])
                
                # Left column (excluding corners, from bottom to top)
                if right > left:
                    for i in range(bottom - 1, top, -1):
                        layer.append(matrix[i][left])
            
            layers.append(layer)
            top += 1
            bottom -= 1
            left += 1
            right -= 1
        
        return layers
    
    # ==========================================
    # 5. SPECIAL TRAVERSAL PATTERNS
    # ==========================================
    
    def knight_tour_traversal(self, matrix: List[List[int]], start_row: int, start_col: int) -> List[int]:
        """Traverse matrix using knight's moves (if possible)
        Time: O(8^(m*n)), Space: O(m*n)
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        visited = [[False] * n for _ in range(m)]
        result = []
        
        # Knight moves: 8 possible moves
        moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                 (1, -2), (1, 2), (2, -1), (2, 1)]
        
        def is_valid(row, col):
            return 0 <= row < m and 0 <= col < n and not visited[row][col]
        
        def dfs(row, col):
            visited[row][col] = True
            result.append(matrix[row][col])
            
            for dr, dc in moves:
                new_row, new_col = row + dr, col + dc
                if is_valid(new_row, new_col):
                    dfs(new_row, new_col)
                    return True
            
            return len(result) == m * n
        
        if is_valid(start_row, start_col):
            dfs(start_row, start_col)
        
        return result
    
    def spiral_traversal_generator(self, matrix: List[List[int]]) -> Generator[int, None, None]:
        """Generator for spiral traversal (memory efficient)
        Time: O(m*n), Space: O(1)
        """
        if not matrix or not matrix[0]:
            return
        
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        
        while top <= bottom and left <= right:
            # Traverse right
            for col in range(left, right + 1):
                yield matrix[top][col]
            top += 1
            
            # Traverse down
            for row in range(top, bottom + 1):
                yield matrix[row][right]
            right -= 1
            
            # Traverse left
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    yield matrix[bottom][col]
                bottom -= 1
            
            # Traverse up
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    yield matrix[row][left]
                left += 1

# Test Examples
def run_examples():
    mtp = MatrixTraversalPatterns()
    
    print("=== MATRIX TRAVERSAL PATTERNS EXAMPLES ===\n")
    
    # Test matrix
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    
    print("Original Matrix:")
    for row in matrix:
        print(row)
    
    # Spiral traversals
    print("\n1. SPIRAL TRAVERSALS:")
    spiral_clockwise = mtp.spiral_traversal_clockwise(matrix)
    print(f"Spiral Clockwise: {spiral_clockwise}")
    
    spiral_counter = mtp.spiral_traversal_counterclockwise(matrix)
    print(f"Spiral Counter-clockwise: {spiral_counter}")
    
    # Generate spiral matrix
    print("\n2. GENERATED SPIRAL MATRIX:")
    spiral_gen = mtp.generate_spiral_matrix(4)
    for row in spiral_gen:
        print(row)
    
    # Diagonal traversals
    print("\n3. DIAGONAL TRAVERSALS:")
    diagonal = mtp.diagonal_traversal_top_to_bottom(matrix)
    print(f"Diagonal traversal: {diagonal}")
    
    main_diagonals = mtp.main_diagonal_traversal(matrix)
    print(f"Main diagonals: {main_diagonals}")
    
    # Zigzag traversals
    print("\n4. ZIGZAG TRAVERSALS:")
    zigzag_row = mtp.zigzag_row_traversal(matrix)
    print(f"Zigzag by rows: {zigzag_row}")
    
    zigzag_col = mtp.zigzag_column_traversal(matrix)
    print(f"Zigzag by columns: {zigzag_col}")
    
    wave = mtp.wave_traversal(matrix)
    print(f"Wave traversal: {wave}")
    
    # Layer-based traversal
    print("\n5. LAYER-BASED TRAVERSAL:")
    layers = mtp.layer_wise_traversal(matrix)
    print(f"Layers: {layers}")
    
    # Generator example
    print("\n6. SPIRAL GENERATOR:")
    print("Spiral using generator:", end=" ")
    for val in mtp.spiral_traversal_generator(matrix):
        print(val, end=" ")
    print()

if __name__ == "__main__":
    run_examples() 