"""
Problem: Rat In Maze
URL: https://practice.geeksforgeeks.org/problems/rat-in-a-maze-problem/1

Problem Statement:
Given NxN maze (0=blocked, 1=open), find all paths from (0,0) to (N-1,N-1). Can move D,L,R,U. Print paths in sorted order.

Sample Input/Output:
Input: N=4, maze[][] = {{1,0,0,0},{1,1,0,1},{1,1,0,0},{0,1,1,1}}
Output: DDRDRR DRDDRR
Explanation: Two paths exist from (0,0) to (3,3)
"""


class Solution:
    def Find_Path_DFS_Backtracking(self, maze, n):
        """
        DFS backtracking with visited array
        Time Complexity: O(4^(n^2))
        Space Complexity: O(n^2)
        """
        result = []
        visited = [[False] * n for _ in range(n)]
        
        if maze[0][0] == 0:
            return result
        
        def dfs(row, col, current_path):
            if row == n - 1 and col == n - 1:
                result.append(current_path)
                return
            
            visited[row][col] = True
            
            directions = [(1, 0), (0, -1), (0, 1), (-1, 0)]
            moves = ['D', 'L', 'R', 'U']
            
            for i in range(4):
                new_row = row + directions[i][0]
                new_col = col + directions[i][1]
                
                if (0 <= new_row < n and 0 <= new_col < n and
                    not visited[new_row][new_col] and maze[new_row][new_col] == 1):
                    dfs(new_row, new_col, current_path + moves[i])
            
            visited[row][col] = False
        
        dfs(0, 0, "")
        result.sort()
        return result
    
    def Count_Paths_DP(self, maze, n):
        """
        DP count paths
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        if maze[0][0] == 0 or maze[n-1][n-1] == 0:
            return 0
        
        dp = [[0] * n for _ in range(n)]
        dp[0][0] = 1
        
        for i in range(n):
            for j in range(n):
                if maze[i][j] == 1:
                    if i > 0:
                        dp[i][j] += dp[i-1][j]
                    if j > 0:
                        dp[i][j] += dp[i][j-1]
        
        return dp[n-1][n-1]


def Test_Rat_In_Maze():
    solution = Solution()
    
    maze1 = [[1,0,0,0],[1,1,0,1],[1,1,0,0],[0,1,1,1]]
    paths = solution.Find_Path_DFS_Backtracking(maze1, 4)
    print("Paths:", " ".join(paths))
    
    count = solution.Count_Paths_DP(maze1, 4)
    print("Total paths count:", count)


if __name__ == "__main__":
    Test_Rat_In_Maze()
