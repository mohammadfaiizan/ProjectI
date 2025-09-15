"""
Problem: Rat in a Maze
URL: https://www.geeksforgeeks.org/problems/rat-in-a-maze-problem/1&selectedLang=python3

Problem Statement:
Consider a rat placed at (0, 0) in a square matrix of size N * N. It has to reach the destination at (N - 1, N - 1). 
Find all possible paths that the rat can take to reach from source to destination. 
The directions in which the rat can move are 'U'(up), 'D'(down), 'L' (left), 'R' (right). 
Value 0 at a cell in the matrix represents that it is blocked and rat cannot move to it while value 1 at a cell in the matrix represents that rat can be travel through it.

Sample Input/Output:
Input: N = 4, m[][] = {{1, 0, 0, 0},
                       {1, 1, 0, 1}, 
                       {1, 1, 0, 0},
                       {0, 1, 1, 1}}
Output: ["DDRDRR", "DRDDRR"]
Explanation: The rat can reach the destination at (3,3) from (0,0) by two paths

Input: N = 2, m[][] = {{1, 0}, 
                       {1, 0}}
Output: []
Explanation: No path exists and destination cell is blocked.
"""

from typing import List

class Solution:
    def Find_Path_Brute_Force(self, m: List[List[int]], n: int) -> List[str]:
        """
        Brute Force - Generate all possible paths
        Time Complexity: O(4^(n²))
        Space Complexity: O(n²)
        """
        if m[0][0] == 0 or m[n-1][n-1] == 0:
            return []
        
        result = []
        visited = [[False] * n for _ in range(n)]
        
        def Generate_All_Paths(x: int, y: int, path: str) -> None:
            if x == n - 1 and y == n - 1:
                result.append(path)
                return
            
            if x < 0 or x >= n or y < 0 or y >= n or m[x][y] == 0 or visited[x][y]:
                return
            
            visited[x][y] = True
            
            Generate_All_Paths(x + 1, y, path + 'D')
            Generate_All_Paths(x - 1, y, path + 'U')
            Generate_All_Paths(x, y + 1, path + 'R')
            Generate_All_Paths(x, y - 1, path + 'L')
            
            visited[x][y] = False
        
        Generate_All_Paths(0, 0, "")
        return sorted(result)
    
    def Find_Path_Backtracking_Optimal(self, m: List[List[int]], n: int) -> List[str]:
        """
        Backtracking Approach - Optimal solution with pruning
        Time Complexity: O(4^(n²))
        Space Complexity: O(n²)
        """
        if m[0][0] == 0 or m[n-1][n-1] == 0:
            return []
        
        result = []
        visited = [[False] * n for _ in range(n)]
        
        def Backtrack(x: int, y: int, path: str) -> None:
            if x == n - 1 and y == n - 1:
                result.append(path)
                return
            
            visited[x][y] = True
            
            directions = [(1, 0, 'D'), (-1, 0, 'U'), (0, 1, 'R'), (0, -1, 'L')]
            
            for dx, dy, direction in directions:
                new_x, new_y = x + dx, y + dy
                
                if (0 <= new_x < n and 0 <= new_y < n and 
                    m[new_x][new_y] == 1 and not visited[new_x][new_y]):
                    Backtrack(new_x, new_y, path + direction)
            
            visited[x][y] = False
        
        Backtrack(0, 0, "")
        return sorted(result)
    
    def Find_Path_DFS_Recursive(self, m: List[List[int]], n: int) -> List[str]:
        """
        DFS Recursive - Depth-first search approach
        Time Complexity: O(4^(n²))
        Space Complexity: O(n²)
        """
        if m[0][0] == 0 or m[n-1][n-1] == 0:
            return []
        
        result = []
        
        def DFS(x: int, y: int, path: str, visited: List[List[bool]]) -> None:
            if x == n - 1 and y == n - 1:
                result.append(path)
                return
            
            if x < 0 or x >= n or y < 0 or y >= n or m[x][y] == 0 or visited[x][y]:
                return
            
            visited[x][y] = True
            
            DFS(x + 1, y, path + 'D', visited)
            DFS(x, y - 1, path + 'L', visited)
            DFS(x, y + 1, path + 'R', visited)
            DFS(x - 1, y, path + 'U', visited)
            
            visited[x][y] = False
        
        visited = [[False] * n for _ in range(n)]
        DFS(0, 0, "", visited)
        return sorted(result)
    
    def Find_Path_Memoized(self, m: List[List[int]], n: int) -> List[str]:
        """
        Memoized Backtracking - Cache intermediate results
        Time Complexity: O(4^(n²))
        Space Complexity: O(n² * 4^(n²))
        """
        if m[0][0] == 0 or m[n-1][n-1] == 0:
            return []
        
        memo = {}
        
        def Get_Paths(x: int, y: int, visited_tuple: tuple) -> List[str]:
            if (x, y, visited_tuple) in memo:
                return memo[(x, y, visited_tuple)]
            
            if x == n - 1 and y == n - 1:
                return [""]
            
            if x < 0 or x >= n or y < 0 or y >= n or m[x][y] == 0 or (x, y) in visited_tuple:
                return []
            
            new_visited = visited_tuple | {(x, y)}
            paths = []
            
            directions = [(1, 0, 'D'), (0, -1, 'L'), (0, 1, 'R'), (-1, 0, 'U')]
            
            for dx, dy, direction in directions:
                sub_paths = Get_Paths(x + dx, y + dy, new_visited)
                for path in sub_paths:
                    paths.append(direction + path)
            
            memo[(x, y, visited_tuple)] = paths
            return paths
        
        result = Get_Paths(0, 0, frozenset())
        return sorted(result)

def Test_Find_Path():
    solution = Solution()
    
    test_cases = [
        ([[1, 0, 0, 0],
          [1, 1, 0, 1], 
          [1, 1, 0, 0],
          [0, 1, 1, 1]], 4, ["DDRDRR", "DRDDRR"]),
        ([[1, 0], 
          [1, 0]], 2, []),
        ([[1, 1], 
          [1, 1]], 2, ["DR", "RD"]),
        ([[1]], 1, [""]),
    ]
    
    for m, n, expected in test_cases:
        result1 = solution.Find_Path_Brute_Force([row.copy() for row in m], n)
        result2 = solution.Find_Path_Backtracking_Optimal([row.copy() for row in m], n)
        result3 = solution.Find_Path_DFS_Recursive([row.copy() for row in m], n)
        
        print(f"Matrix: {m}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Backtracking Optimal: {result2}")
        print(f"DFS Recursive: {result3}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Path()
