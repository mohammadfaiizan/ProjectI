class MatrixTraversal:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)  # assuming square matrix (n x n)
    
    # 1. Spiral Matrix Traversal
    def spiral_traversal(self):
        '''
        Problem Statement:
        Given an N x N matrix, traverse the matrix in a spiral order starting from the top-left corner.

        Approach:
        - Use four boundaries (top, bottom, left, right) and iterate while adjusting the boundaries.
        
        Time Complexity: O(n^2), as we visit every element in the matrix exactly once.
        Space Complexity: O(1), as we are not using extra space apart from variables for the boundaries.
        '''
        result = []
        top, bottom, left, right = 0, self.n - 1, 0, self.n - 1
        
        while top <= bottom and left <= right:
            # Traverse from left to right
            for i in range(left, right + 1):
                result.append(self.matrix[top][i])
            top += 1

            # Traverse from top to bottom
            for i in range(top, bottom + 1):
                result.append(self.matrix[i][right])
            right -= 1

            if top <= bottom:
                # Traverse from right to left
                for i in range(right, left - 1, -1):
                    result.append(self.matrix[bottom][i])
                bottom -= 1

            if left <= right:
                # Traverse from bottom to top
                for i in range(bottom, top - 1, -1):
                    result.append(self.matrix[i][left])
                left += 1
                
        return result


    # 2. Diagonal Traversal of a Matrix
    def diagonal_traversal(self):
        '''
        Problem Statement:
        Given an N x N matrix, traverse the matrix diagonally.

        Approach:
        - Traverse the matrix diagonally by iterating through diagonals and adding elements to result.
        
        Time Complexity: O(n^2), as we visit every element of the matrix once.
        Space Complexity: O(n^2), as we are storing the result in a list.
        '''
        result = []
        
        # Traverse diagonals starting from the first row and first column
        for d in range(2 * self.n - 1):
            diagonal = []
            # Get elements in the current diagonal
            row = 0 if d < self.n else d - self.n + 1
            col = d if d < self.n else self.n - 1
            
            while row < self.n and col >= 0:
                diagonal.append(self.matrix[row][col])
                row += 1
                col -= 1
            
            result.extend(diagonal)
        
        return result


    # 3. Zigzag (Snake) Traversal of a Matrix
    def zigzag_traversal(self):
        '''
        Problem Statement:
        Given an N x N matrix, traverse the matrix in a zigzag (snake-like) fashion.

        Approach:
        - For even rows, traverse from left to right and for odd rows, traverse from right to left.
        
        Time Complexity: O(n^2), as we visit every element of the matrix once.
        Space Complexity: O(1), if we don't store the traversal and just print.
        '''
        result = []
        for i in range(self.n):
            if i % 2 == 0:
                result.extend(self.matrix[i])
            else:
                result.extend(self.matrix[i][::-1])
        return result


    # 4. Matrix Search (Element Search in a Sorted Matrix)
    def search_matrix(self, target):
        '''
        Problem Statement:
        Given an N x N matrix where every row and column is sorted, search for a target element in the matrix.

        Approach:
        - Start from the top-right corner of the matrix and use the fact that rows and columns are sorted to decide whether to move left or down.
        
        Time Complexity: O(n), as we may move at most `n` steps in either direction (left or down).
        Space Complexity: O(1), as we only use pointers to traverse the matrix.
        '''
        row, col = 0, self.n - 1
        
        while row < self.n and col >= 0:
            if self.matrix[row][col] == target:
                return True
            elif self.matrix[row][col] < target:
                row += 1
            else:
                col -= 1
        
        return False


    # 5. Level Order Traversal of Matrix (Row by Row)
    def level_order_traversal(self):
        '''
        Problem Statement:
        Given an N x N matrix, traverse it level by level (like BFS in graphs).

        Approach:
        - Iterate through each row and append to the result list row by row.
        
        Time Complexity: O(n^2), as we traverse every element in the matrix.
        Space Complexity: O(n^2), as we store all the matrix elements.
        '''
        result = []
        for row in range(self.n):
            result.append(self.matrix[row])
        return result


    # 6. Diagonal Sum of Matrix
    def diagonal_sum(self):
        '''
        Problem Statement:
        Find the sum of elements on the diagonals (main diagonal and anti-diagonal).

        Approach:
        - Traverse both the main diagonal and anti-diagonal and sum the elements.
        
        Time Complexity: O(n), as we only visit the diagonal elements.
        Space Complexity: O(1), as we only store the sum.
        '''
        total_sum = 0
        for i in range(self.n):
            total_sum += self.matrix[i][i]  # Main diagonal
            total_sum += self.matrix[i][self.n - 1 - i]  # Anti diagonal
            
        # If the matrix size is odd, we have double-counted the center element
        if self.n % 2 != 0:
            total_sum -= self.matrix[self.n // 2][self.n // 2]
        
        return total_sum


    # 7. Row and Column Sum of Matrix
    def row_column_sum(self):
        '''
        Problem Statement:
        Find the sum of each row and each column in the matrix.

        Approach:
        - Iterate over each row and each column and compute the sums.
        
        Time Complexity: O(n^2), as we traverse every element.
        Space Complexity: O(n), as we store row and column sums.
        '''
        row_sums = []
        col_sums = [0] * self.n
        
        # Compute row sums and column sums
        for row in range(self.n):
            row_sum = sum(self.matrix[row])
            row_sums.append(row_sum)
            for col in range(self.n):
                col_sums[col] += self.matrix[row][col]
        
        return row_sums, col_sums


    # 8. Matrix Rotation (90 degrees clockwise)
    def rotate_matrix(self):
        '''
        Problem Statement:
        Rotate the matrix by 90 degrees clockwise.

        Approach:
        - First, transpose the matrix (swap rows and columns).
        - Then reverse each row of the transposed matrix.
        
        Time Complexity: O(n^2), as we visit each element twice (transpose and reverse).
        Space Complexity: O(1), in-place rotation.
        '''
        # Transpose the matrix
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.matrix[i][j], self.matrix[j][i] = self.matrix[j][i], self.matrix[i][j]
        
        # Reverse each row
        for row in self.matrix:
            row.reverse()


    # 9. Boundary Traversal of Matrix
    def boundary_traversal(self):
        '''
        Problem Statement:
        Traverse the boundary of a matrix in clockwise order.

        Approach:
        - Traverse the first row, last column, last row, and first column while ensuring no corner elements are repeated.
        
        Time Complexity: O(n), as we only visit the boundary elements.
        Space Complexity: O(1), as we do not store any additional data.
        '''
        result = []
        
        # Traverse first row
        for col in range(self.n):
            result.append(self.matrix[0][col])
        
        # Traverse last column
        for row in range(1, self.n):
            result.append(self.matrix[row][self.n - 1])
        
        # Traverse last row
        if self.n > 1:
            for col in range(self.n - 2, -1, -1):
                result.append(self.matrix[self.n - 1][col])
        
        # Traverse first column
        if self.n > 1:
            for row in range(self.n - 2, 0, -1):
                result.append(self.matrix[row][0])
        
        return result


    # 10. Transpose of a Matrix
    def transpose_matrix(self):
        '''
        Problem Statement:
        Given an N x N matrix, find the transpose of the matrix.

        Approach:
        - Transpose the matrix in place by swapping elements (matrix[i][j] with matrix[j][i]).
        
        Time Complexity: O(n^2), as we visit each element once.
        Space Complexity: O(1), in-place transposition.
        '''
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Swap matrix[i][j] with matrix[j][i]
                self.matrix[i][j], self.matrix[j][i] = self.matrix[j][i], self.matrix[i][j]

# Example Usage
if __name__ == "__main__":
    mat = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    mt = MatrixTraversal(mat)
    print("Spiral Traversal:", mt.spiral_traversal())
    print("Diagonal Traversal:", mt.diagonal_traversal())
    print("Zigzag Traversal:", mt.zigzag_traversal())
    print("Matrix Search (5):", mt.search_matrix(5))
    print("Level Order Traversal:", mt.level_order_traversal())
    print("Diagonal Sum:", mt.diagonal_sum())
    print("Row and Column Sum:", mt.row_column_sum())
    
    mt.rotate_matrix()
    print("Rotated Matrix:", mt.matrix)
    
    print("Boundary Traversal:", mt.boundary_traversal())
    
    mt.transpose_matrix()
    print("Transposed Matrix:", mt.matrix)
