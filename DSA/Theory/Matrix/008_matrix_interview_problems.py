"""
Matrix Interview Problems
========================

Topics: Common matrix problems asked in interviews
Companies: Google, Facebook, Amazon, Microsoft, Apple, Netflix
Difficulty: Medium to Hard
"""

from typing import List, Tuple
from collections import deque

class MatrixInterviewProblems:
    
    # ==========================================
    # 1. GOOGLE PROBLEMS
    # ==========================================
    
    def word_search(self, board: List[List[str]], word: str) -> bool:
        """LC 79: Word Search (Google, Facebook)"""
        if not board or not board[0]:
            return False
        
        m, n = len(board), len(board[0])
        
        def dfs(i, j, index):
            if index == len(word):
                return True
            
            if (i < 0 or i >= m or j < 0 or j >= n or 
                board[i][j] != word[index]):
                return False
            
            # Mark as visited
            temp = board[i][j]
            board[i][j] = '#'
            
            # Explore all 4 directions
            found = (dfs(i+1, j, index+1) or 
                    dfs(i-1, j, index+1) or 
                    dfs(i, j+1, index+1) or 
                    dfs(i, j-1, index+1))
            
            # Restore original value
            board[i][j] = temp
            
            return found
        
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        
        return False
    
    # ==========================================
    # 2. FACEBOOK/META PROBLEMS
    # ==========================================
    
    def walls_and_gates(self, rooms: List[List[int]]) -> None:
        """LC 286: Walls and Gates (Facebook)"""
        if not rooms or not rooms[0]:
            return
        
        m, n = len(rooms), len(rooms[0])
        queue = deque()
        INF = 2147483647
        
        # Find all gates
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    queue.append((i, j))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col = queue.popleft()
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    rooms[new_row][new_col] == INF):
                    rooms[new_row][new_col] = rooms[row][col] + 1
                    queue.append((new_row, new_col))
    
    # ==========================================
    # 3. AMAZON PROBLEMS
    # ==========================================
    
    def treasure_island(self, grid: List[List[str]]) -> int:
        """Amazon: Find minimum steps to reach treasure"""
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        queue = deque([(0, 0, 0)])  # row, col, steps
        visited = set([(0, 0)])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col, steps = queue.popleft()
            
            if grid[row][col] == 'X':  # Found treasure
                return steps
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    (new_row, new_col) not in visited and 
                    grid[new_row][new_col] != 'D'):  # Not blocked
                    
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, steps + 1))
        
        return -1
    
    # ==========================================
    # 4. MICROSOFT PROBLEMS
    # ==========================================
    
    def pacific_atlantic_water_flow(self, heights: List[List[int]]) -> List[List[int]]:
        """LC 417: Pacific Atlantic Water Flow (Microsoft)"""
        if not heights or not heights[0]:
            return []
        
        m, n = len(heights), len(heights[0])
        pacific = [[False] * n for _ in range(m)]
        atlantic = [[False] * n for _ in range(m)]
        
        def dfs(i, j, ocean):
            ocean[i][j] = True
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and 
                    not ocean[ni][nj] and 
                    heights[ni][nj] >= heights[i][j]):
                    dfs(ni, nj, ocean)
        
        # DFS from Pacific borders
        for i in range(m):
            dfs(i, 0, pacific)
        for j in range(n):
            dfs(0, j, pacific)
        
        # DFS from Atlantic borders
        for i in range(m):
            dfs(i, n - 1, atlantic)
        for j in range(n):
            dfs(m - 1, j, atlantic)
        
        # Find cells that can reach both oceans
        result = []
        for i in range(m):
            for j in range(n):
                if pacific[i][j] and atlantic[i][j]:
                    result.append([i, j])
        
        return result

# Test Examples
def run_examples():
    mip = MatrixInterviewProblems()
    
    print("=== MATRIX INTERVIEW PROBLEMS ===\n")
    
    # Word Search
    print("1. WORD SEARCH (Google):")
    board = [
        ['A','B','C','E'],
        ['S','F','C','S'],
        ['A','D','E','E']
    ]
    word = "ABCCED"
    
    found = mip.word_search(board, word)
    print(f"Word '{word}' found: {found}")

if __name__ == "__main__":
    run_examples() 