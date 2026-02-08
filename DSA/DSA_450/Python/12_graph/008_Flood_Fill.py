"""
Problem: Flood Fill Algorithm
URL: https://leetcode.com/problems/flood-fill/

Problem Statement:
Given an image (2D grid), a starting pixel, and new color, perform flood fill.

Sample Input/Output:
Input: image=[[1,1,1],[1,1,0],[1,0,1]], sr=1, sc=1, color=2
Output: [[2,2,2],[2,2,0],[2,0,1]]
"""

from collections import deque


class Solution:
    def Flood_Fill_DFS_Helper(self, row, col, oldColor, newColor, image, m, n):
        if (row < 0 or row >= m or col < 0 or col >= n or 
            image[row][col] != oldColor or image[row][col] == newColor):
            return
        
        image[row][col] = newColor
        
        self.Flood_Fill_DFS_Helper(row + 1, col, oldColor, newColor, image, m, n)
        self.Flood_Fill_DFS_Helper(row - 1, col, oldColor, newColor, image, m, n)
        self.Flood_Fill_DFS_Helper(row, col + 1, oldColor, newColor, image, m, n)
        self.Flood_Fill_DFS_Helper(row, col - 1, oldColor, newColor, image, m, n)

    def Flood_Fill_DFS(self, image, sr, sc, color):
        """
        Recursive DFS
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m = len(image)
        n = len(image[0])
        oldColor = image[sr][sc]
        
        if oldColor != color:
            self.Flood_Fill_DFS_Helper(sr, sc, oldColor, color, image, m, n)
        
        return image

    def Flood_Fill_BFS(self, image, sr, sc, color):
        """
        Iterative BFS
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m = len(image)
        n = len(image[0])
        oldColor = image[sr][sc]
        
        if oldColor == color:
            return image
        
        q = deque()
        q.append((sr, sc))
        image[sr][sc] = color
        
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while q:
            row, col = q.popleft()
            
            for dx, dy in directions:
                newRow = row + dx
                newCol = col + dy
                
                if (0 <= newRow < m and 0 <= newCol < n and 
                    image[newRow][newCol] == oldColor):
                    image[newRow][newCol] = color
                    q.append((newRow, newCol))
        
        return image


def Test_Flood_Fill():
    solution = Solution()
    
    print("Test: Flood Fill")
    image = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1]
    ]
    
    print("Original image:")
    for row in image:
        for pixel in row:
            print(pixel, end=" ")
        print()
    
    import copy
    result1 = solution.Flood_Fill_DFS(copy.deepcopy(image), 1, 1, 2)
    print("\nAfter flood fill (DFS):")
    for row in result1:
        for pixel in row:
            print(pixel, end=" ")
        print()
    
    result2 = solution.Flood_Fill_BFS(copy.deepcopy(image), 1, 1, 2)
    print("\nAfter flood fill (BFS):")
    for row in result2:
        for pixel in row:
            print(pixel, end=" ")
        print()


if __name__ == "__main__":
    Test_Flood_Fill()
