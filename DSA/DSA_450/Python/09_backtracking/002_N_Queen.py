"""
Problem: N Queen
URL: https://www.geeksforgeeks.org/printing-solutions-n-queen-problem/

Problem Statement:
Place N queens on NxN board so no two attack each other. Print all solutions.

Sample Input/Output:
Input: N=4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: Two distinct solutions exist for 4-queen problem
"""


class Solution:
    def Solve_N_Queen_Backtracking(self, n):
        """
        Backtracking with isSafe check
        Time Complexity: O(N!)
        Space Complexity: O(N^2)
        """
        result = []
        board = [['.'] * n for _ in range(n)]
        
        def Is_Safe(row, col, board):
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
        
        def backtrack(row):
            if row == n:
                result.append([''.join(row) for row in board])
                return
            
            for col in range(n):
                if Is_Safe(row, col, board):
                    board[row][col] = 'Q'
                    backtrack(row + 1)
                    board[row][col] = '.'
        
        backtrack(0)
        return result
    
    def Solve_N_Queen_Optimized(self, n):
        """
        Optimized with row/diagonal arrays
        Time Complexity: O(N!)
        Space Complexity: O(N)
        """
        result = []
        board = [['.'] * n for _ in range(n)]
        col_used = [False] * n
        diag1 = [False] * (2 * n - 1)
        diag2 = [False] * (2 * n - 1)
        
        def backtrack(row):
            if row == n:
                result.append([''.join(row) for row in board])
                return
            
            for col in range(n):
                d1 = row + col
                d2 = row - col + n - 1
                
                if not col_used[col] and not diag1[d1] and not diag2[d2]:
                    board[row][col] = 'Q'
                    col_used[col] = diag1[d1] = diag2[d2] = True
                    backtrack(row + 1)
                    col_used[col] = diag1[d1] = diag2[d2] = False
                    board[row][col] = '.'
        
        backtrack(0)
        return result


def Test_N_Queen():
    solution = Solution()
    
    n = 4
    solutions = solution.Solve_N_Queen_Backtracking(n)
    print("Number of solutions:", len(solutions))
    
    for i, sol in enumerate(solutions):
        print(f"Solution {i + 1}:")
        for row in sol:
            print(row)
        print()


if __name__ == "__main__":
    Test_N_Queen()
