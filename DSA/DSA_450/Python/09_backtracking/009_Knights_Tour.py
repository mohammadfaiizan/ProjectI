"""
Problem: Knights Tour
URL: https://www.geeksforgeeks.org/the-knights-tour-problem-backtracking-1/

Problem Statement:
Find a knight's tour on NxN chessboard - visit every square exactly once.

Sample Input/Output:
Input: N=8, start=(0,0)
Output: 8x8 matrix with move numbers
Explanation: Knight visits all 64 squares exactly once
"""


class Solution:
    def Knights_Tour_Backtracking(self, n, start_row, start_col):
        """
        Backtracking with 8 moves
        Time Complexity: O(8^(N^2))
        Space Complexity: O(N^2)
        """
        board = [[-1] * n for _ in range(n)]
        
        moves = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
        
        def Is_Safe(row, col, move_num):
            return 0 <= row < n and 0 <= col < n and board[row][col] == -1
        
        def backtrack(row, col, move_num):
            board[row][col] = move_num
            
            if move_num == n * n - 1:
                return True
            
            for dx, dy in moves:
                new_row = row + dx
                new_col = col + dy
                
                if Is_Safe(new_row, new_col, move_num + 1):
                    if backtrack(new_row, new_col, move_num + 1):
                        return True
            
            board[row][col] = -1
            return False
        
        if backtrack(start_row, start_col, 0):
            return board
        
        return []


def Test_Knights_Tour():
    solution = Solution()
    
    n = 8
    tour = solution.Knights_Tour_Backtracking(n, 0, 0)
    
    if tour:
        print("Knight's Tour found:")
        for i in range(n):
            print(" ".join(f"{tour[i][j]:3d}" for j in range(n)))
    else:
        print("No solution found")


if __name__ == "__main__":
    Test_Knights_Tour()
