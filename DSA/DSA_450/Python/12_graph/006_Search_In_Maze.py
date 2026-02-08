"""
Problem: Rat in a Maze / Search in Maze
URL: https://practice.geeksforgeeks.org/problems/rat-in-a-maze-problem/1

Problem Statement:
Given a maze (N x N matrix with 0s and 1s), find all paths from (0,0) to (N-1,N-1). Can move in all 4 directions (D,L,R,U).

Sample Input/Output:
Input: 4x4 maze with blocked cells
Output: All valid paths as direction strings
"""


class Solution:
    def Search_Maze_Backtracking_Helper(self, row, col, n, maze, result, path, visited):
        if row == n - 1 and col == n - 1:
            result.append(path)
            return
        
        directions = [(1, 0), (0, -1), (0, 1), (-1, 0)]
        dirChars = ['D', 'L', 'R', 'U']
        
        for i in range(4):
            newRow = row + directions[i][0]
            newCol = col + directions[i][1]
            
            if (0 <= newRow < n and 0 <= newCol < n and 
                maze[newRow][newCol] == 1 and not visited[newRow][newCol]):
                visited[newRow][newCol] = True
                self.Search_Maze_Backtracking_Helper(newRow, newCol, n, maze, result, path + dirChars[i], visited)
                visited[newRow][newCol] = False

    def Search_Maze_Backtracking(self, n, maze):
        """
        DFS Backtracking - All Paths
        Time Complexity: O(4^(n^2))
        Space Complexity: O(n^2)
        """
        result = []
        path = ""
        visited = [[False] * n for _ in range(n)]
        
        if maze[0][0] == 1:
            visited[0][0] = True
            self.Search_Maze_Backtracking_Helper(0, 0, n, maze, result, path, visited)
        
        return result

    def Search_Maze_Single_Path_Helper(self, row, col, n, maze, path, visited):
        if row == n - 1 and col == n - 1:
            return True
        
        directions = [(1, 0), (0, -1), (0, 1), (-1, 0)]
        dirChars = ['D', 'L', 'R', 'U']
        
        for i in range(4):
            newRow = row + directions[i][0]
            newCol = col + directions[i][1]
            
            if (0 <= newRow < n and 0 <= newCol < n and 
                maze[newRow][newCol] == 1 and not visited[newRow][newCol]):
                visited[newRow][newCol] = True
                path.append(dirChars[i])
                
                if self.Search_Maze_Single_Path_Helper(newRow, newCol, n, maze, path, visited):
                    return True
                
                path.pop()
                visited[newRow][newCol] = False
        
        return False

    def Search_Maze_Single_Path(self, n, maze):
        """
        Find Any One Path using DFS
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        path = []
        visited = [[False] * n for _ in range(n)]
        
        if maze[0][0] == 1:
            visited[0][0] = True
            if self.Search_Maze_Single_Path_Helper(0, 0, n, maze, path, visited):
                return ''.join(path)
        
        return ""


def Test_Search_In_Maze():
    solution = Solution()
    
    print("Test: 4x4 Maze")
    n = 4
    maze = [
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 1]
    ]
    
    allPaths = solution.Search_Maze_Backtracking(n, maze)
    print(f"All paths found: {len(allPaths)}")
    for path in allPaths:
        print(path, end=" ")
    print()
    
    singlePath = solution.Search_Maze_Single_Path(n, maze)
    print(f"\nSingle path: {singlePath if singlePath else 'No path'}")


if __name__ == "__main__":
    Test_Search_In_Maze()
