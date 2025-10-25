"""
Problem: Capture the Pawns
URL: https://www.naukri.com/code360/problems/capture-the-pawns

Problem Statement:
Given a chessboard represented as a 2D grid, count how many pawns can be captured by a rook 
placed at position (r, c). A rook can capture pawns in the same row or column.

Pawns are represented by 'P', empty cells by '.', and the rook position by 'R'.

Sample Input/Output:
Input: board = [[".",".","."],["P","R","P"],[".",".","."]],  r = 1, c = 1
Output: 2
Explanation: Rook at (1,1) can capture both pawns in its row

Input: board = [["P",".","."],[".","R","."],[".",".","."]],  r = 1, c = 1
Output: 1

Input: board = [[".",".","."],[".",".","."],[".","R","."]],  r = 2, c = 1
Output: 0
"""

from typing import List

class Solution:
    def Capture_Pawns_Brute_Force(self, board: List[List[str]], r: int, c: int) -> int:
        """
        Brute Force Approach - Check all four directions
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        count = 0
        m, n = len(board), len(board[0])
        
        for i in range(m):
            if i != r and board[i][c] == 'P':
                count += 1
        
        for j in range(n):
            if j != c and board[r][j] == 'P':
                count += 1
        
        return count
    
    def Capture_Pawns_Four_Directions(self, board: List[List[str]], r: int, c: int) -> int:
        """
        Four Directions Approach - Stop at first pawn
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        count = 0
        m, n = len(board), len(board[0])
        
        for i in range(r - 1, -1, -1):
            if board[i][c] == 'P':
                count += 1
                break
        
        for i in range(r + 1, m):
            if board[i][c] == 'P':
                count += 1
                break
        
        for j in range(c - 1, -1, -1):
            if board[r][j] == 'P':
                count += 1
                break
        
        for j in range(c + 1, n):
            if board[r][j] == 'P':
                count += 1
                break
        
        return count
    
    def Capture_Pawns_DFS(self, board: List[List[str]], r: int, c: int) -> int:
        """
        DFS Approach in Four Directions
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        count = 0
        m, n = len(board), len(board[0])
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            while 0 <= nr < m and 0 <= nc < n:
                if board[nr][nc] == 'P':
                    count += 1
                    break
                nr += dr
                nc += dc
        
        return count
    
    def Capture_Pawns_Optimized(self, board: List[List[str]], r: int, c: int) -> int:
        """
        Optimized Approach
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        m, n = len(board), len(board[0])
        count = 0
        
        count += sum(1 for i in range(m) if board[i][c] == 'P')
        count += sum(1 for j in range(n) if board[r][j] == 'P')
        
        return count
    
    def Capture_Pawns_List_Comprehension(self, board: List[List[str]], r: int, c: int) -> int:
        """
        List Comprehension Approach
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        m, n = len(board), len(board[0])
        
        row_pawns = sum(board[r][j] == 'P' for j in range(n))
        col_pawns = sum(board[i][c] == 'P' for i in range(m))
        
        return row_pawns + col_pawns

def Test_Capture_Pawns():
    solution = Solution()
    
    test_cases = [
        ([[".",".","."],["P","R","P"],[".",".","."]],  1, 1, 2),
        ([["P",".","."],[".","R","."],[".",".","."]],  1, 1, 1),
        ([[".",".","."],[".",".","."],[".","R","."]],  2, 1, 0),
        ([["P",".","."],["P","R","."],["P",".","."]],  1, 1, 3),
        ([["R","P"]], 0, 0, 1)
    ]
    
    for board, r, c, expected in test_cases:
        result1 = solution.Capture_Pawns_Brute_Force([row[:] for row in board], r, c)
        result2 = solution.Capture_Pawns_Four_Directions([row[:] for row in board], r, c)
        result3 = solution.Capture_Pawns_DFS([row[:] for row in board], r, c)
        result4 = solution.Capture_Pawns_Optimized([row[:] for row in board], r, c)
        result5 = solution.Capture_Pawns_List_Comprehension([row[:] for row in board], r, c)
        
        print(f"Board: {board}, Rook at ({r},{c})")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Four Directions: {result2}")
        print(f"DFS: {result3}")
        print(f"Optimized: {result4}")
        print(f"List Comprehension: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Capture_Pawns()

