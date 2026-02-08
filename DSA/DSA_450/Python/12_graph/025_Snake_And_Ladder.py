"""
Problem: Snakes and Ladders
URL: https://leetcode.com/problems/snakes-and-ladders/

Problem Statement:
Given a snakes and ladders board, find the minimum number of dice throws required to reach the end of the board. The board is represented as a 2D array where -1 means no snake or ladder, and a positive number means a snake or ladder that takes you to that cell.

Sample Input/Output:
Input: board = [[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,35,-1,-1,13,-1],[-1,-1,-1,-1,-1,-1],[-1,15,-1,-1,-1,-1]]
Output: 4
"""

from collections import deque


class Solution:
    def Snake_Ladder_BFS(self, board):
        """
        BFS from cell 1, handle snakes/ladders as edges
        Time Complexity: O(N)
        Space Complexity: O(N)
        """
        n = len(board)
        total = n * n
        dist = [-1] * (total + 1)
        q = deque()
        
        dist[1] = 0
        q.append(1)
        
        while q:
            curr = q.popleft()
            
            if curr == total:
                return dist[curr]
            
            for dice in range(1, 7):
                if curr + dice > total:
                    break
                next_cell = curr + dice
                row = n - 1 - (next_cell - 1) // n
                col = (next_cell - 1) % n
                if (n - 1 - row) % 2 == 1:
                    col = n - 1 - col
                
                if board[row][col] != -1:
                    next_cell = board[row][col]
                
                if dist[next_cell] == -1:
                    dist[next_cell] = dist[curr] + 1
                    q.append(next_cell)
        
        return -1


def Test_Snake_Ladder_BFS():
    solution = Solution()
    
    board1 = [
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, 35, -1, -1, 13, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, 15, -1, -1, -1, -1]
    ]
    print(f"Test 1: {solution.Snake_Ladder_BFS(board1)}")
    
    board2 = [
        [-1, -1],
        [-1, 3]
    ]
    print(f"Test 2: {solution.Snake_Ladder_BFS(board2)}")
    
    board3 = [
        [-1, 1, 2, -1],
        [2, 13, 15, -1],
        [-1, 10, -1, -1],
        [-1, 6, 2, 8]
    ]
    print(f"Test 3: {solution.Snake_Ladder_BFS(board3)}")


if __name__ == "__main__":
    Test_Snake_Ladder_BFS()
