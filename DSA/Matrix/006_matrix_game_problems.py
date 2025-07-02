"""
Matrix Game Problems
===================

Topics: Game of Life, cellular automata, matrix state transitions
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List
from collections import defaultdict

class MatrixGameProblems:
    
    # ==========================================
    # 1. GAME OF LIFE
    # ==========================================
    
    def game_of_life(self, board: List[List[int]]) -> None:
        """LC 289: Game of Life - in-place solution"""
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        
        # Count live neighbors
        def count_live_neighbors(row, col):
            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < m and 0 <= new_col < n:
                        # Use & 1 to get original state (before encoding)
                        count += board[new_row][new_col] & 1
            return count
        
        # First pass: encode next state
        for i in range(m):
            for j in range(n):
                live_neighbors = count_live_neighbors(i, j)
                
                # Apply Game of Life rules
                if board[i][j] == 1:  # Currently alive
                    if live_neighbors == 2 or live_neighbors == 3:
                        board[i][j] = 3  # 11 in binary (was alive, stays alive)
                else:  # Currently dead
                    if live_neighbors == 3:
                        board[i][j] = 2  # 10 in binary (was dead, becomes alive)
        
        # Second pass: decode next state
        for i in range(m):
            for j in range(n):
                board[i][j] >>= 1  # Get the second bit
    
    def game_of_life_infinite(self, live_cells: List[List[int]]) -> List[List[int]]:
        """Game of Life on infinite board"""
        live_set = set(map(tuple, live_cells))
        
        # Count neighbors for all relevant cells
        neighbor_count = defaultdict(int)
        
        for x, y in live_set:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_count[(x + dx, y + dy)] += 1
        
        next_generation = []
        
        for (x, y), count in neighbor_count.items():
            if (x, y) in live_set:
                # Currently alive
                if count == 2 or count == 3:
                    next_generation.append([x, y])
            else:
                # Currently dead
                if count == 3:
                    next_generation.append([x, y])
        
        return next_generation
    
    # ==========================================
    # 2. TIC-TAC-TOE
    # ==========================================
    
    def check_tic_tac_toe_winner(self, board: List[List[str]]) -> str:
        """Check winner in Tic-Tac-Toe"""
        n = len(board)
        
        # Check rows
        for i in range(n):
            if all(board[i][j] == board[i][0] and board[i][0] != ' ' for j in range(n)):
                return board[i][0]
        
        # Check columns
        for j in range(n):
            if all(board[i][j] == board[0][j] and board[0][j] != ' ' for i in range(n)):
                return board[0][j]
        
        # Check main diagonal
        if all(board[i][i] == board[0][0] and board[0][0] != ' ' for i in range(n)):
            return board[0][0]
        
        # Check anti-diagonal
        if all(board[i][n-1-i] == board[0][n-1] and board[0][n-1] != ' ' for i in range(n)):
            return board[0][n-1]
        
        return "Draw" if all(board[i][j] != ' ' for i in range(n) for j in range(n)) else "Ongoing"
    
    def design_tic_tac_toe(self, n: int):
        """LC 348: Design Tic-Tac-Toe"""
        class TicTacToe:
            def __init__(self, n: int):
                self.n = n
                self.rows = [0] * n
                self.cols = [0] * n
                self.diagonal = 0
                self.anti_diagonal = 0
            
            def move(self, row: int, col: int, player: int) -> int:
                move_val = 1 if player == 1 else -1
                
                self.rows[row] += move_val
                self.cols[col] += move_val
                
                if row == col:
                    self.diagonal += move_val
                
                if row + col == self.n - 1:
                    self.anti_diagonal += move_val
                
                # Check for winner
                if (abs(self.rows[row]) == self.n or 
                    abs(self.cols[col]) == self.n or 
                    abs(self.diagonal) == self.n or 
                    abs(self.anti_diagonal) == self.n):
                    return player
                
                return 0
        
        return TicTacToe(n)
    
    # ==========================================
    # 3. MINESWEEPER
    # ==========================================
    
    def minesweeper(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        """LC 529: Minesweeper"""
        m, n = len(board), len(board[0])
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        def dfs(row, col):
            if row < 0 or row >= m or col < 0 or col >= n or board[row][col] != 'E':
                return
            
            # Count adjacent mines
            mine_count = 0
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < m and 0 <= new_col < n and 
                    board[new_row][new_col] == 'M'):
                    mine_count += 1
            
            if mine_count > 0:
                board[row][col] = str(mine_count)
            else:
                board[row][col] = 'B'
                # Recursively reveal adjacent cells
                for dr, dc in directions:
                    dfs(row + dr, col + dc)
        
        row, col = click[0], click[1]
        
        if board[row][col] == 'M':
            board[row][col] = 'X'
        else:
            dfs(row, col)
        
        return board
    
    # ==========================================
    # 4. SUDOKU SOLVER
    # ==========================================
    
    def solve_sudoku(self, board: List[List[str]]) -> None:
        """LC 37: Sudoku Solver"""
        def is_valid(board, row, col, num):
            # Check row
            for j in range(9):
                if board[row][j] == num:
                    return False
            
            # Check column
            for i in range(9):
                if board[i][col] == num:
                    return False
            
            # Check 3x3 box
            start_row, start_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    if board[i][j] == num:
                        return False
            
            return True
        
        def backtrack():
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in '123456789':
                            if is_valid(board, i, j, num):
                                board[i][j] = num
                                
                                if backtrack():
                                    return True
                                
                                board[i][j] = '.'
                        
                        return False
            return True
        
        backtrack()
    
    def is_valid_sudoku(self, board: List[List[str]]) -> bool:
        """LC 36: Valid Sudoku"""
        seen = set()
        
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = board[i][j]
                    box_id = (i // 3) * 3 + j // 3
                    
                    if (f"row{i}-{num}" in seen or 
                        f"col{j}-{num}" in seen or 
                        f"box{box_id}-{num}" in seen):
                        return False
                    
                    seen.add(f"row{i}-{num}")
                    seen.add(f"col{j}-{num}")
                    seen.add(f"box{box_id}-{num}")
        
        return True

# Test Examples
def run_examples():
    mgp = MatrixGameProblems()
    
    print("=== MATRIX GAME PROBLEMS ===\n")
    
    # Game of Life
    print("1. GAME OF LIFE:")
    board = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 0]
    ]
    
    print("Original board:")
    for row in board:
        print(row)
    
    mgp.game_of_life(board)
    print("\nAfter one generation:")
    for row in board:
        print(row)
    
    # Tic-Tac-Toe
    print("\n2. TIC-TAC-TOE:")
    tic_tac_toe = [
        ['X', 'O', 'X'],
        ['O', 'X', 'O'],
        ['O', 'X', 'X']
    ]
    
    winner = mgp.check_tic_tac_toe_winner(tic_tac_toe)
    print(f"Winner: {winner}")
    
    # Sudoku validation
    print("\n3. SUDOKU VALIDATION:")
    sudoku = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    
    is_valid = mgp.is_valid_sudoku(sudoku)
    print(f"Is valid Sudoku: {is_valid}")

if __name__ == "__main__":
    run_examples() 