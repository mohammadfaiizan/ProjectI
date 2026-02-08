"""
Problem: Minimum Steps by a Knight
URL: https://practice.geeksforgeeks.org/problems/steps-by-knight5927/1

Problem Statement:
Find minimum steps for a knight to reach from source to target on an N x N chessboard.

Sample Input/Output:
Input: N=6, source=(4,5), target=(1,1)
Output: Minimum steps: 3
"""

from collections import deque


class Solution:
    def Min_Steps_Knight_BFS(self, N, source, target):
        """
        BFS Exploring All 8 Knight Moves
        Time Complexity: O(N^2)
        Space Complexity: O(N^2)
        """
        if source[0] == target[0] and source[1] == target[1]:
            return 0
        
        visited = [[False] * N for _ in range(N)]
        q = deque()
        
        dx = [-2, -1, 1, 2, 2, 1, -1, -2]
        dy = [1, 2, 2, 1, -1, -2, -2, -1]
        
        visited[source[0]][source[1]] = True
        q.append((source[0], source[1], 0))
        
        while q:
            x, y, steps = q.popleft()
            
            for i in range(8):
                newX = x + dx[i]
                newY = y + dy[i]
                
                if 0 <= newX < N and 0 <= newY < N and not visited[newX][newY]:
                    if newX == target[0] and newY == target[1]:
                        return steps + 1
                    
                    visited[newX][newY] = True
                    q.append((newX, newY, steps + 1))
        
        return -1


def Test_Minimum_Steps_Knight():
    solution = Solution()
    
    print("Test 1: N=6, source=(4,5), target=(1,1)")
    steps1 = solution.Min_Steps_Knight_BFS(6, (4, 5), (1, 1))
    print(f"Minimum steps: {steps1}")
    
    print("\nTest 2: N=8, source=(0,0), target=(7,7)")
    steps2 = solution.Min_Steps_Knight_BFS(8, (0, 0), (7, 7))
    print(f"Minimum steps: {steps2}")
    
    print("\nTest 3: N=5, source=(0,0), target=(4,4)")
    steps3 = solution.Min_Steps_Knight_BFS(5, (0, 0), (4, 4))
    print(f"Minimum steps: {steps3}")


if __name__ == "__main__":
    Test_Minimum_Steps_Knight()
