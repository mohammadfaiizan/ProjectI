"""
Problem: Flood Fill
URL: https://leetcode.com/problems/flood-fill/description/

Problem Statement:
An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image.
You are also given three integers sr, sc, and color. You should perform a flood fill on the image starting from the pixel image[sr][sc].
To perform a flood fill, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with color.
Return the modified image after performing the flood fill.

Sample Input/Output:
Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
Explanation: From the center of the image with position (sr, sc) = (1, 1) (i.e., the red pixel), all pixels connected by a path of the same color as the starting pixel (i.e., the blue pixels) are colored with the new color.

Input: image = [[0,0,0],[0,0,0]], sr = 0, sc = 0, color = 0
Output: [[0,0,0],[0,0,0]]
Explanation: The starting pixel is already colored 0, which is the same as the target color. Therefore, no change is made to the image.
"""

from typing import List
from collections import deque

class Solution:
    def Flood_Fill_DFS_Recursive(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        DFS Recursive - Classic recursive flood fill
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if not image or sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]):
            return image
        
        original_color = image[sr][sc]
        if original_color == color:
            return image
        
        def DFS(r: int, c: int) -> None:
            if (r < 0 or r >= len(image) or c < 0 or c >= len(image[0]) or 
                image[r][c] != original_color):
                return
            
            image[r][c] = color
            
            DFS(r + 1, c)
            DFS(r - 1, c)
            DFS(r, c + 1)
            DFS(r, c - 1)
        
        DFS(sr, sc)
        return image
    
    def Flood_Fill_BFS_Queue(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        BFS Queue - Use queue for level-wise filling
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if not image or sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]):
            return image
        
        original_color = image[sr][sc]
        if original_color == color:
            return image
        
        m, n = len(image), len(image[0])
        queue = deque([(sr, sc)])
        image[sr][sc] = color
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < m and 0 <= nc < n and 
                    image[nr][nc] == original_color):
                    image[nr][nc] = color
                    queue.append((nr, nc))
        
        return image
    
    def Flood_Fill_DFS_Iterative(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        DFS Iterative - Use stack for DFS
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if not image or sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]):
            return image
        
        original_color = image[sr][sc]
        if original_color == color:
            return image
        
        m, n = len(image), len(image[0])
        stack = [(sr, sc)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while stack:
            r, c = stack.pop()
            
            if (0 <= r < m and 0 <= c < n and image[r][c] == original_color):
                image[r][c] = color
                
                for dr, dc in directions:
                    stack.append((r + dr, c + dc))
        
        return image
    
    def Flood_Fill_Union_Find(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        Union Find - Use disjoint set for connected components
        Time Complexity: O(m * n * α(m * n))
        Space Complexity: O(m * n)
        """
        if not image or sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]):
            return image
        
        original_color = image[sr][sc]
        if original_color == color:
            return image
        
        m, n = len(image), len(image[0])
        
        class UnionFind:
            def __init__(self, size: int):
                self.parent = list(range(size))
                self.rank = [0] * size
            
            def Find(self, x: int) -> int:
                if self.parent[x] != x:
                    self.parent[x] = self.Find(self.parent[x])
                return self.parent[x]
            
            def Union(self, x: int, y: int) -> None:
                root_x, root_y = self.Find(x), self.Find(y)
                if root_x != root_y:
                    if self.rank[root_x] < self.rank[root_y]:
                        self.parent[root_x] = root_y
                    elif self.rank[root_x] > self.rank[root_y]:
                        self.parent[root_y] = root_x
                    else:
                        self.parent[root_y] = root_x
                        self.rank[root_x] += 1
        
        uf = UnionFind(m * n)
        
        for i in range(m):
            for j in range(n):
                if image[i][j] == original_color:
                    for di, dj in [(0, 1), (1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < m and 0 <= nj < n and 
                            image[ni][nj] == original_color):
                            uf.Union(i * n + j, ni * n + nj)
        
        target_root = uf.Find(sr * n + sc)
        
        for i in range(m):
            for j in range(n):
                if (image[i][j] == original_color and 
                    uf.Find(i * n + j) == target_root):
                    image[i][j] = color
        
        return image
    
    def Flood_Fill_Boundary_Fill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        """
        Boundary Fill - Alternative flood fill with boundary checking
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        if not image or sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]):
            return image
        
        original_color = image[sr][sc]
        if original_color == color:
            return image
        
        def Fill(r: int, c: int, boundary_color: int, fill_color: int) -> None:
            if (r < 0 or r >= len(image) or c < 0 or c >= len(image[0]) or 
                image[r][c] == boundary_color or image[r][c] == fill_color):
                return
            
            if image[r][c] == original_color:
                image[r][c] = fill_color
                Fill(r + 1, c, boundary_color, fill_color)
                Fill(r - 1, c, boundary_color, fill_color)
                Fill(r, c + 1, boundary_color, fill_color)
                Fill(r, c - 1, boundary_color, fill_color)
        
        Fill(sr, sc, -1, color)
        return image

def Test_Flood_Fill():
    solution = Solution()
    
    test_cases = [
        ([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2, [[2,2,2],[2,2,0],[2,0,1]]),
        ([[0,0,0],[0,0,0]], 0, 0, 0, [[0,0,0],[0,0,0]]),
        ([[0,0,0],[0,1,1]], 1, 1, 1, [[0,0,0],[0,1,1]]),
        ([[1]], 0, 0, 2, [[2]]),
        ([[1,1,1],[1,1,1],[1,1,1]], 1, 1, 3, [[3,3,3],[3,3,3],[3,3,3]])
    ]
    
    methods = [
        ("DFS Recursive", solution.Flood_Fill_DFS_Recursive),
        ("BFS Queue", solution.Flood_Fill_BFS_Queue),
        ("DFS Iterative", solution.Flood_Fill_DFS_Iterative),
        ("Union Find", solution.Flood_Fill_Union_Find),
        ("Boundary Fill", solution.Flood_Fill_Boundary_Fill)
    ]
    
    for image, sr, sc, color, expected in test_cases:
        print(f"Image: {image}")
        print(f"Start: ({sr}, {sc}), Color: {color}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            image_copy = [row.copy() for row in image]
            result = method(image_copy, sr, sc, color)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Flood_Fill()
