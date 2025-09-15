"""
Problem: N Queens
URL: https://leetcode.com/problems/n-queens/

Problem Statement:
The n-queens puzzle is the problem of placing n chess queens on an n×n chessboard so that no two queens attack each other.
Given an integer n, return all distinct solutions to the n-queens puzzle.
Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

Sample Input/Output:
Input: n = 4
Output: [[".Q..",
          "...Q",
          "Q...",
          "..Q."],
         ["..Q.",
          "Q...",
          "...Q",
          ".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle

Input: n = 1
Output: [["Q"]]
Explanation: Only one solution for 1-queen
"""

from typing import List

class Solution:
    def Solve_N_Queens_Brute_Force(self, n: int) -> List[List[str]]:
        """
        Brute Force - Check all possible placements
        Time Complexity: O(n^(n²))
        Space Complexity: O(n²)
        """
        def Is_Safe(board: List[List[str]], row: int, col: int) -> bool:
            for i in range(row):
                if board[i][col] == 'Q':
                    return False
            
            i, j = row - 1, col - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            
            i, j = row - 1, col + 1
            while i >= 0 and j < n:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            
            return True
        
        def Solve(board: List[List[str]], row: int, solutions: List[List[str]]) -> None:
            if row == n:
                solutions.append([''.join(board[i]) for i in range(n)])
                return
            
            for col in range(n):
                if Is_Safe(board, row, col):
                    board[row][col] = 'Q'
                    Solve(board, row + 1, solutions)
                    board[row][col] = '.'
        
        board = [['.' for _ in range(n)] for _ in range(n)]
        solutions = []
        Solve(board, 0, solutions)
        return solutions
    
    def Solve_N_Queens_Backtracking_Optimal(self, n: int) -> List[List[str]]:
        """
        Backtracking Optimal - Efficient conflict detection
        Time Complexity: O(n!)
        Space Complexity: O(n)
        """
        result = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        
        def Is_Safe(row: int, col: int) -> bool:
            for i in range(row):
                if board[i][col] == 'Q':
                    return False
                
                if col - (row - i) >= 0 and board[i][col - (row - i)] == 'Q':
                    return False
                
                if col + (row - i) < n and board[i][col + (row - i)] == 'Q':
                    return False
            
            return True
        
        def Backtrack(row: int) -> None:
            if row == n:
                result.append([''.join(board[i]) for i in range(n)])
                return
            
            for col in range(n):
                if Is_Safe(row, col):
                    board[row][col] = 'Q'
                    Backtrack(row + 1)
                    board[row][col] = '.'
        
        Backtrack(0)
        return result
    
    def Solve_N_Queens_Set_Based(self, n: int) -> List[List[str]]:
        """
        Set Based Tracking - Use sets for conflict detection
        Time Complexity: O(n!)
        Space Complexity: O(n)
        """
        result = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        cols = set()
        diag1 = set()
        diag2 = set()
        
        def Backtrack(row: int) -> None:
            if row == n:
                result.append([''.join(board[i]) for i in range(n)])
                return
            
            for col in range(n):
                if col in cols or (row - col) in diag1 or (row + col) in diag2:
                    continue
                
                board[row][col] = 'Q'
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)
                
                Backtrack(row + 1)
                
                board[row][col] = '.'
                cols.remove(col)
                diag1.remove(row - col)
                diag2.remove(row + col)
        
        Backtrack(0)
        return result
    
    def Solve_N_Queens_Bit_Manipulation(self, n: int) -> List[List[str]]:
        """
        Bit Manipulation - Use bits for conflict tracking
        Time Complexity: O(n!)
        Space Complexity: O(n)
        """
        result = []
        
        def Backtrack(row: int, cols: int, diag1: int, diag2: int, board: List[str]) -> None:
            if row == n:
                result.append(board[:])
                return
            
            available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
            
            while available:
                col = available & -available
                col_idx = (col - 1).bit_length() - 1
                
                board.append('.' * col_idx + 'Q' + '.' * (n - col_idx - 1))
                
                Backtrack(row + 1, cols | col, (diag1 | col) << 1, (diag2 | col) >> 1, board)
                
                board.pop()
                available &= available - 1
        
        Backtrack(0, 0, 0, 0, [])
        return result

def Test_Solve_N_Queens():
    solution = Solution()
    
    test_cases = [1, 4, 8]
    
    for n in test_cases:
        result1 = solution.Solve_N_Queens_Brute_Force(n)
        result2 = solution.Solve_N_Queens_Backtracking_Optimal(n)
        result3 = solution.Solve_N_Queens_Set_Based(n)
        result4 = solution.Solve_N_Queens_Bit_Manipulation(n)
        
        print(f"N = {n}")
        print(f"Brute Force solutions: {len(result1)}")
        print(f"Backtracking solutions: {len(result2)}")
        print(f"Set Based solutions: {len(result3)}")
        print(f"Bit Manipulation solutions: {len(result4)}")
        
        if n <= 4:
            print(f"Sample solution:")
            if result2:
                for row in result2[0]:
                    print(f"  {row}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Solve_N_Queens()
