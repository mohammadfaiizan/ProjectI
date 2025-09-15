"""
Problem: Snakes and Ladders
URL: https://leetcode.com/problems/snakes-and-ladders/description/

Problem Statement:
You are given an n x n integer matrix board where the cells are labeled from 1 to n² in a Boustrophedon style starting from the bottom left of the board (i.e. board[n - 1][0]) and alternating direction each row.
You start on square 1 of the board. In each move, you start from square curr and do the following:
- Choose a destination square next with a label in the range [curr + 1, min(curr + 6, n²)].
- If next has a snake or ladder, you must move to the destination of that snake or ladder. Otherwise, you move to next.
The game ends when you reach square n².
Return the least number of moves required to reach square n², or -1 if it is not possible.

Sample Input/Output:
Input: board = [[-1,-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,-1,14,-1,-1,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,35,-1,-1,13,-1]]
Output: 4
Explanation: You start at square 1. From square 1, you can move to squares 2 to 6. You decide to move to square 2. From square 2, you can move to squares 3 to 7. You decide to move to square 6. From square 6, you can move to squares 7 to 12. You decide to move to square 10. From square 10, you can move to squares 11 to 15. You decide to move to square 14. From square 14, you can move to squares 15 to 19. You decide to move to square 20. From square 20, you can move to squares 21 to 26. You decide to move to square 25. From square 25, you can move to squares 26 to 30. You decide to move to square 30. From square 30, you can move to squares 31 to 35. You decide to move to square 35. From square 35, you can move to squares 36 to 36. You decide to move to square 36, which is the end of the board.
"""

from typing import List, Tuple
from collections import deque

class Solution:
    def Get_Board_Position(self, num: int, n: int) -> Tuple[int, int]:
        """
        Get Board Position - Convert number to board coordinates
        """
        num -= 1
        row = n - 1 - num // n
        col = num % n if (n - 1 - row) % 2 == 0 else n - 1 - num % n
        return row, col
    
    def Snakes_Ladders_BFS_Basic(self, board: List[List[int]]) -> int:
        """
        BFS Basic - Standard BFS with dice roll simulation
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        n = len(board)
        target = n * n
        
        queue = deque([(1, 0)])
        visited = {1}
        
        while queue:
            curr_pos, moves = queue.popleft()
            
            if curr_pos == target:
                return moves
            
            for next_pos in range(curr_pos + 1, min(curr_pos + 7, target + 1)):
                row, col = self.Get_Board_Position(next_pos, n)
                
                if board[row][col] != -1:
                    final_pos = board[row][col]
                else:
                    final_pos = next_pos
                
                if final_pos not in visited:
                    visited.add(final_pos)
                    queue.append((final_pos, moves + 1))
        
        return -1
    
    def Snakes_Ladders_BFS_Optimized(self, board: List[List[int]]) -> int:
        """
        BFS Optimized - Pre-process snakes and ladders
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        n = len(board)
        target = n * n
        
        snakes_ladders = {}
        for i in range(n):
            for j in range(n):
                if board[i][j] != -1:
                    num = (n - 1 - i) * n + (j if (n - 1 - i) % 2 == 0 else n - 1 - j) + 1
                    snakes_ladders[num] = board[i][j]
        
        queue = deque([(1, 0)])
        visited = {1}
        
        while queue:
            curr_pos, moves = queue.popleft()
            
            if curr_pos == target:
                return moves
            
            for dice in range(1, 7):
                next_pos = curr_pos + dice
                if next_pos > target:
                    break
                
                final_pos = snakes_ladders.get(next_pos, next_pos)
                
                if final_pos not in visited:
                    visited.add(final_pos)
                    queue.append((final_pos, moves + 1))
        
        return -1
    
    def Snakes_Ladders_Dijkstra(self, board: List[List[int]]) -> int:
        """
        Dijkstra Algorithm - Use priority queue for shortest path
        Time Complexity: O(n² log n²)
        Space Complexity: O(n²)
        """
        import heapq
        
        n = len(board)
        target = n * n
        
        dist = [float('inf')] * (target + 1)
        dist[1] = 0
        pq = [(0, 1)]
        
        while pq:
            curr_moves, curr_pos = heapq.heappop(pq)
            
            if curr_pos == target:
                return curr_moves
            
            if curr_moves > dist[curr_pos]:
                continue
            
            for dice in range(1, 7):
                next_pos = curr_pos + dice
                if next_pos > target:
                    break
                
                row, col = self.Get_Board_Position(next_pos, n)
                final_pos = board[row][col] if board[row][col] != -1 else next_pos
                
                new_moves = curr_moves + 1
                if new_moves < dist[final_pos]:
                    dist[final_pos] = new_moves
                    heapq.heappush(pq, (new_moves, final_pos))
        
        return -1
    
    def Snakes_Ladders_DP_Memoization(self, board: List[List[int]]) -> int:
        """
        DP Memoization - Recursive with memoization
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        n = len(board)
        target = n * n
        memo = {}
        
        def Min_Moves(pos: int) -> int:
            if pos == target:
                return 0
            
            if pos in memo:
                return memo[pos]
            
            min_moves = float('inf')
            
            for dice in range(1, 7):
                next_pos = pos + dice
                if next_pos > target:
                    break
                
                row, col = self.Get_Board_Position(next_pos, n)
                final_pos = board[row][col] if board[row][col] != -1 else next_pos
                
                moves = Min_Moves(final_pos)
                if moves != float('inf'):
                    min_moves = min(min_moves, moves + 1)
            
            memo[pos] = min_moves
            return min_moves
        
        result = Min_Moves(1)
        return result if result != float('inf') else -1
    
    def Snakes_Ladders_Bottom_Up_DP(self, board: List[List[int]]) -> int:
        """
        Bottom Up DP - Build solution from target backwards
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        n = len(board)
        target = n * n
        
        dp = [float('inf')] * (target + 1)
        dp[target] = 0
        
        for pos in range(target - 1, 0, -1):
            for dice in range(1, 7):
                next_pos = pos + dice
                if next_pos > target:
                    break
                
                row, col = self.Get_Board_Position(next_pos, n)
                final_pos = board[row][col] if board[row][col] != -1 else next_pos
                
                if dp[final_pos] != float('inf'):
                    dp[pos] = min(dp[pos], dp[final_pos] + 1)
        
        return dp[1] if dp[1] != float('inf') else -1

def Test_Snakes_Ladders():
    solution = Solution()
    
    test_cases = [
        ([[-1,-1,-1,-1,-1,-1],
          [-1,-1,-1,-1,-1,-1],
          [-1,-1,-1,-1,-1,-1],
          [-1,-1,14,-1,-1,-1],
          [-1,-1,-1,-1,-1,-1],
          [-1,35,-1,-1,13,-1]], 4),
        ([[-1,-1],
          [-1,3]], 1),
        ([[-1,-1,-1],
          [-1,9,-1],
          [-1,-1,-1]], 1)
    ]
    
    methods = [
        ("BFS Basic", solution.Snakes_Ladders_BFS_Basic),
        ("BFS Optimized", solution.Snakes_Ladders_BFS_Optimized),
        ("Dijkstra", solution.Snakes_Ladders_Dijkstra),
        ("DP Memoization", solution.Snakes_Ladders_DP_Memoization),
        ("Bottom Up DP", solution.Snakes_Ladders_Bottom_Up_DP)
    ]
    
    for board, expected in test_cases:
        print(f"Board:")
        for row in board:
            print(f"  {row}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            board_copy = [row.copy() for row in board]
            result = method(board_copy)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Snakes_Ladders()
