"""
Problem: Sudoku Solver
URL: https://leetcode.com/problems/sudoku-solver/description/

Problem Statement:
Write a program to solve a Sudoku puzzle by filling the empty cells.
A sudoku solution must satisfy all of the following rules:
1. Each of the digits 1-9 must occur exactly once in each row.
2. Each of the digits 1-9 must occur exactly once in each column.
3. Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
The '.' character indicates empty cells.

Sample Input/Output:
Input: board = [["5","3",".",".","7",".",".",".","."],
                ["6",".",".","1","9","5",".",".","."],
                [".","9","8",".",".",".",".","6","."],
                ["8",".",".",".","6",".",".",".","3"],
                ["4",".",".","8",".","3",".",".","1"],
                ["7",".",".",".","2",".",".",".","6"],
                [".","6",".",".",".",".","2","8","."],
                [".",".",".","4","1","9",".",".","5"],
                [".",".",".",".","8",".",".","7","9"]]
Output: [["5","3","4","6","7","8","9","1","2"],
         ["6","7","2","1","9","5","3","4","8"],
         ["1","9","8","3","4","2","5","6","7"],
         ["8","5","9","7","6","1","4","2","3"],
         ["4","2","6","8","5","3","7","9","1"],
         ["7","1","3","9","2","4","8","5","6"],
         ["9","6","1","5","3","7","2","8","4"],
         ["2","8","7","4","1","9","6","3","5"],
         ["3","4","5","2","8","6","1","7","9"]]
"""

from typing import List

class Solution:
    def Solve_Sudoku_Brute_Force(self, board: List[List[str]]) -> None:
        """
        Brute Force - Try all possibilities
        Time Complexity: O(9^(n²))
        Space Complexity: O(n²)
        """
        def Is_Valid(board: List[List[str]], row: int, col: int, num: str) -> bool:
            for i in range(9):
                if board[row][i] == num or board[i][col] == num:
                    return False
            
            start_row, start_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    if board[i][j] == num:
                        return False
            
            return True
        
        def Solve(board: List[List[str]]) -> bool:
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in '123456789':
                            if Is_Valid(board, i, j, num):
                                board[i][j] = num
                                if Solve(board):
                                    return True
                                board[i][j] = '.'
                        return False
            return True
        
        Solve(board)
    
    def Solve_Sudoku_Backtracking_Optimal(self, board: List[List[str]]) -> None:
        """
        Backtracking Optimal - Efficient constraint checking
        Time Complexity: O(9^(n²))
        Space Complexity: O(n²)
        """
        def Is_Valid(row: int, col: int, num: str) -> bool:
            for i in range(9):
                if board[row][i] == num or board[i][col] == num:
                    return False
            
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    if board[i][j] == num:
                        return False
            
            return True
        
        def Backtrack() -> bool:
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in '123456789':
                            if Is_Valid(i, j, num):
                                board[i][j] = num
                                if Backtrack():
                                    return True
                                board[i][j] = '.'
                        return False
            return True
        
        Backtrack()
    
    def Solve_Sudoku_Set_Based(self, board: List[List[str]]) -> None:
        """
        Set Based Tracking - Use sets for constraint checking
        Time Complexity: O(9^(n²))
        Space Complexity: O(n²)
        """
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = board[i][j]
                    rows[i].add(num)
                    cols[j].add(num)
                    boxes[3 * (i // 3) + j // 3].add(num)
        
        def Backtrack() -> bool:
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        box_idx = 3 * (i // 3) + j // 3
                        for num in '123456789':
                            if num not in rows[i] and num not in cols[j] and num not in boxes[box_idx]:
                                board[i][j] = num
                                rows[i].add(num)
                                cols[j].add(num)
                                boxes[box_idx].add(num)
                                
                                if Backtrack():
                                    return True
                                
                                board[i][j] = '.'
                                rows[i].remove(num)
                                cols[j].remove(num)
                                boxes[box_idx].remove(num)
                        return False
            return True
        
        Backtrack()
    
    def Solve_Sudoku_Most_Constrained_First(self, board: List[List[str]]) -> None:
        """
        Most Constrained First - Choose cell with fewest possibilities
        Time Complexity: O(9^(n²))
        Space Complexity: O(n²)
        """
        def Get_Possibilities(row: int, col: int) -> List[str]:
            if board[row][col] != '.':
                return []
            
            used = set()
            
            for i in range(9):
                used.add(board[row][i])
                used.add(board[i][col])
            
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    used.add(board[i][j])
            
            return [num for num in '123456789' if num not in used]
        
        def Find_Best_Cell() -> tuple:
            min_possibilities = 10
            best_cell = None
            
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        possibilities = Get_Possibilities(i, j)
                        if len(possibilities) < min_possibilities:
                            min_possibilities = len(possibilities)
                            best_cell = (i, j, possibilities)
            
            return best_cell
        
        def Backtrack() -> bool:
            cell_info = Find_Best_Cell()
            if cell_info is None:
                return True
            
            row, col, possibilities = cell_info
            
            for num in possibilities:
                board[row][col] = num
                if Backtrack():
                    return True
                board[row][col] = '.'
            
            return False
        
        Backtrack()

def Test_Solve_Sudoku():
    solution = Solution()
    
    test_board = [["5","3",".",".","7",".",".",".","."],
                  ["6",".",".","1","9","5",".",".","."],
                  [".","9","8",".",".",".",".","6","."],
                  ["8",".",".",".","6",".",".",".","3"],
                  ["4",".",".","8",".","3",".",".","1"],
                  ["7",".",".",".","2",".",".",".","6"],
                  [".","6",".",".",".",".","2","8","."],
                  [".",".",".","4","1","9",".",".","5"],
                  [".",".",".",".","8",".",".","7","9"]]
    
    methods = [
        ("Brute Force", solution.Solve_Sudoku_Brute_Force),
        ("Backtracking Optimal", solution.Solve_Sudoku_Backtracking_Optimal),
        ("Set Based", solution.Solve_Sudoku_Set_Based),
        ("Most Constrained First", solution.Solve_Sudoku_Most_Constrained_First)
    ]
    
    for method_name, method in methods:
        board_copy = [row.copy() for row in test_board]
        print(f"Testing {method_name}:")
        
        method(board_copy)
        
        print("Solved board:")
        for row in board_copy:
            print(" ".join(row))
        print("-" * 50)

if __name__ == "__main__":
    Test_Solve_Sudoku()
