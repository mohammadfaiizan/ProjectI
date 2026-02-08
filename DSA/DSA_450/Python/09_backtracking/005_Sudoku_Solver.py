"""
Problem: Sudoku Solver
URL: https://practice.geeksforgeeks.org/problems/solve-the-sudoku-1587115621/1

Problem Statement:
Solve a 9x9 Sudoku puzzle using backtracking. Fill empty cells (0) with digits 1-9.

Sample Input/Output:
Input: grid[][] = {{3,0,6,5,0,8,4,0,0},{5,2,0,0,0,0,0,0,0},{0,8,7,0,0,0,0,3,1},{0,0,3,0,1,0,0,8,0},{9,0,0,8,6,3,0,0,5},{0,5,0,0,9,0,6,0,0},{1,3,0,0,0,0,2,5,0},{0,0,0,0,0,0,0,7,4},{0,0,5,2,0,6,3,0,0}}
Output: Solved grid
Explanation: Fill all zeros with valid digits 1-9
"""


class Solution:
    def Solve_Sudoku_Backtracking(self, grid):
        """
        Backtracking with row/col/box validation
        Time Complexity: O(9^(n*n))
        Space Complexity: O(n*n)
        """
        def Is_Safe(row, col, num):
            for i in range(9):
                if grid[row][i] == num or grid[i][col] == num:
                    return False
            
            box_row = (row // 3) * 3
            box_col = (col // 3) * 3
            for i in range(3):
                for j in range(3):
                    if grid[box_row + i][box_col + j] == num:
                        return False
            
            return True
        
        def solve():
            for row in range(9):
                for col in range(9):
                    if grid[row][col] == 0:
                        for num in range(1, 10):
                            if Is_Safe(row, col, num):
                                grid[row][col] = num
                                if solve():
                                    return True
                                grid[row][col] = 0
                        return False
            return True
        
        return solve()


def Test_Sudoku_Solver():
    solution = Solution()
    
    grid = [
        [3,0,6,5,0,8,4,0,0],
        [5,2,0,0,0,0,0,0,0],
        [0,8,7,0,0,0,0,3,1],
        [0,0,3,0,1,0,0,8,0],
        [9,0,0,8,6,3,0,0,5],
        [0,5,0,0,9,0,6,0,0],
        [1,3,0,0,0,0,2,5,0],
        [0,0,0,0,0,0,0,7,4],
        [0,0,5,2,0,6,3,0,0]
    ]
    
    solved = solution.Solve_Sudoku_Backtracking(grid)
    
    if solved:
        print("Sudoku solved:")
        for i in range(9):
            print(" ".join(str(grid[i][j]) for j in range(9)))
    else:
        print("No solution exists")


if __name__ == "__main__":
    Test_Sudoku_Solver()
