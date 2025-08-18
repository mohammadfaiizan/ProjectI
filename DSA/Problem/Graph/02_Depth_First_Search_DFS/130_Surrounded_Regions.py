"""
130. Surrounded Regions
Difficulty: Medium

Problem:
Given an m x n matrix board containing 'X' and 'O', capture all regions that are 
4-directionally surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

Examples:
Input: board = [["X","X","X","X"],
                ["X","O","O","X"],
                ["X","X","O","X"],
                ["X","O","X","X"]]
Output: [["X","X","X","X"],
         ["X","X","X","X"],
         ["X","X","X","X"],
         ["X","O","X","X"]]

Input: board = [["X"]]
Output: [["X"]]

Constraints:
- m == board.length
- n == board[i].length
- 1 <= m, n <= 200
- board[i][j] is 'X' or 'O'
"""

from typing import List
from collections import deque

class Solution:
    def solve_approach1_boundary_dfs(self, board: List[List[str]]) -> None:
        """
        Approach 1: Boundary DFS (Optimal Strategy)
        
        Key insight: 'O' regions connected to boundary cannot be captured.
        1. Mark all boundary-connected 'O' regions as safe
        2. Capture all remaining 'O' regions
        
        Time: O(M*N) - visit each cell at most twice
        Space: O(M*N) - recursion stack depth
        """
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        
        def dfs_mark_safe(i, j):
            """Mark all connected 'O' cells as safe ('S')"""
            if (i < 0 or i >= m or j < 0 or j >= n or 
                board[i][j] != 'O'):
                return
            
            board[i][j] = 'S'  # Mark as safe
            
            # Explore 4 directions
            dfs_mark_safe(i + 1, j)
            dfs_mark_safe(i - 1, j)
            dfs_mark_safe(i, j + 1)
            dfs_mark_safe(i, j - 1)
        
        # Step 1: Mark all boundary-connected 'O' regions as safe
        # Top and bottom rows
        for j in range(n):
            if board[0][j] == 'O':
                dfs_mark_safe(0, j)
            if board[m-1][j] == 'O':
                dfs_mark_safe(m-1, j)
        
        # Left and right columns
        for i in range(m):
            if board[i][0] == 'O':
                dfs_mark_safe(i, 0)
            if board[i][n-1] == 'O':
                dfs_mark_safe(i, n-1)
        
        # Step 2: Process entire board
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'  # Capture surrounded region
                elif board[i][j] == 'S':
                    board[i][j] = 'O'  # Restore safe region
    
    def solve_approach2_boundary_bfs(self, board: List[List[str]]) -> None:
        """
        Approach 2: Boundary BFS
        
        Same strategy as DFS but using BFS for marking safe regions.
        
        Time: O(M*N)
        Space: O(M*N) - queue size
        """
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def bfs_mark_safe(start_i, start_j):
            """Mark all connected 'O' cells as safe using BFS"""
            if board[start_i][start_j] != 'O':
                return
            
            queue = deque([(start_i, start_j)])
            board[start_i][start_j] = 'S'
            
            while queue:
                i, j = queue.popleft()
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and board[ni][nj] == 'O'):
                        board[ni][nj] = 'S'
                        queue.append((ni, nj))
        
        # Mark boundary-connected regions as safe
        for j in range(n):
            bfs_mark_safe(0, j)
            bfs_mark_safe(m-1, j)
        
        for i in range(m):
            bfs_mark_safe(i, 0)
            bfs_mark_safe(i, n-1)
        
        # Capture surrounded regions and restore safe ones
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'S':
                    board[i][j] = 'O'
    
    def solve_approach3_union_find(self, board: List[List[str]]) -> None:
        """
        Approach 3: Union-Find approach
        
        Connect all 'O' cells, including a virtual boundary node.
        Cells connected to boundary are safe.
        
        Time: O(M*N*α(M*N))
        Space: O(M*N)
        """
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        
        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [0] * size
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
        
        # Create UF with extra node for boundary
        boundary_node = m * n
        uf = UnionFind(m * n + 1)
        
        def get_id(i, j):
            return i * n + j
        
        # Connect boundary 'O' cells to boundary node
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    cell_id = get_id(i, j)
                    
                    # Connect boundary cells to boundary node
                    if i == 0 or i == m-1 or j == 0 or j == n-1:
                        uf.union(cell_id, boundary_node)
                    
                    # Connect to adjacent 'O' cells
                    for di, dj in [(1, 0), (0, 1)]:  # Only right and down
                        ni, nj = i + di, j + dj
                        if (ni < m and nj < n and board[ni][nj] == 'O'):
                            uf.union(cell_id, get_id(ni, nj))
        
        # Capture cells not connected to boundary
        for i in range(m):
            for j in range(n):
                if (board[i][j] == 'O' and 
                    uf.find(get_id(i, j)) != uf.find(boundary_node)):
                    board[i][j] = 'X'
    
    def solve_approach4_iterative_dfs(self, board: List[List[str]]) -> None:
        """
        Approach 4: Iterative DFS to avoid recursion limits
        
        Use explicit stack for DFS traversal.
        
        Time: O(M*N)
        Space: O(M*N) - stack size
        """
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def iterative_dfs_mark_safe(start_i, start_j):
            """Mark connected 'O' cells as safe using iterative DFS"""
            if board[start_i][start_j] != 'O':
                return
            
            stack = [(start_i, start_j)]
            
            while stack:
                i, j = stack.pop()
                
                if (i < 0 or i >= m or j < 0 or j >= n or 
                    board[i][j] != 'O'):
                    continue
                
                board[i][j] = 'S'  # Mark as safe
                
                # Add neighbors to stack
                for di, dj in directions:
                    stack.append((i + di, j + dj))
        
        # Mark boundary-connected regions as safe
        for j in range(n):
            iterative_dfs_mark_safe(0, j)
            iterative_dfs_mark_safe(m-1, j)
        
        for i in range(m):
            iterative_dfs_mark_safe(i, 0)
            iterative_dfs_mark_safe(i, n-1)
        
        # Final processing
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'S':
                    board[i][j] = 'O'

def test_surrounded_regions():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (board, expected)
        ([["X","X","X","X"],
          ["X","O","O","X"],
          ["X","X","O","X"],
          ["X","O","X","X"]], 
         [["X","X","X","X"],
          ["X","X","X","X"],
          ["X","X","X","X"],
          ["X","O","X","X"]]),
        ([["X"]], [["X"]]),
        ([["O"]], [["O"]]),  # Boundary 'O' stays
        ([["O","O"],
          ["O","O"]], 
         [["O","O"],
          ["O","O"]]),  # All boundary connected
        ([["X","O","X"],
          ["O","X","O"],
          ["X","O","X"]], 
         [["X","O","X"],
          ["O","X","O"],
          ["X","O","X"]]),  # All boundary connected
    ]
    
    approaches = [
        ("Boundary DFS", solution.solve_approach1_boundary_dfs),
        ("Boundary BFS", solution.solve_approach2_boundary_bfs),
        ("Union-Find", solution.solve_approach3_union_find),
        ("Iterative DFS", solution.solve_approach4_iterative_dfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (board, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_board = [row[:] for row in board]
            func(test_board)
            status = "✓" if test_board == expected else "✗"
            print(f"Test {i+1}: {status}")
            print(f"         Input:    {board}")
            print(f"         Expected: {expected}")
            print(f"         Got:      {test_board}")

def demonstrate_capture_process():
    """Demonstrate the region capture process"""
    print("\n=== Region Capture Process Demo ===")
    
    board = [
        ["X","X","X","X"],
        ["X","O","O","X"],
        ["X","X","O","X"],
        ["X","O","X","X"]
    ]
    
    print("Original board:")
    for i, row in enumerate(board):
        print(f"  Row {i}: {row}")
    
    # Create working copy
    work_board = [row[:] for row in board]
    m, n = len(work_board), len(work_board[0])
    
    print(f"\nStep 1: Mark boundary-connected 'O' regions as safe")
    
    def dfs_mark_safe(i, j, step_num):
        if (i < 0 or i >= m or j < 0 or j >= n or work_board[i][j] != 'O'):
            return
        
        work_board[i][j] = 'S'
        print(f"  Marked ({i},{j}) as safe")
        
        dfs_mark_safe(i + 1, j, step_num)
        dfs_mark_safe(i - 1, j, step_num)
        dfs_mark_safe(i, j + 1, step_num)
        dfs_mark_safe(i, j - 1, step_num)
    
    # Check boundary cells
    boundary_starts = []
    for j in range(n):
        if work_board[0][j] == 'O':
            boundary_starts.append((0, j, "top"))
        if work_board[m-1][j] == 'O':
            boundary_starts.append((m-1, j, "bottom"))
    
    for i in range(m):
        if work_board[i][0] == 'O':
            boundary_starts.append((i, 0, "left"))
        if work_board[i][n-1] == 'O':
            boundary_starts.append((i, n-1, "right"))
    
    print(f"  Boundary 'O' cells found: {[(i, j, pos) for i, j, pos in boundary_starts]}")
    
    for i, j, position in boundary_starts:
        print(f"  Starting DFS from boundary cell ({i},{j}) at {position}")
        dfs_mark_safe(i, j, 1)
    
    print(f"\nBoard after marking safe regions:")
    for i, row in enumerate(work_board):
        print(f"  Row {i}: {row}")
    
    print(f"\nStep 2: Capture remaining 'O' regions and restore safe ones")
    for i in range(m):
        for j in range(n):
            if work_board[i][j] == 'O':
                print(f"  Capturing ({i},{j}): O -> X")
                work_board[i][j] = 'X'
            elif work_board[i][j] == 'S':
                print(f"  Restoring ({i},{j}): S -> O")
                work_board[i][j] = 'O'
    
    print(f"\nFinal board:")
    for i, row in enumerate(work_board):
        print(f"  Row {i}: {row}")

def visualize_surrounded_regions():
    """Create visual representation of surrounded regions"""
    print("\n=== Surrounded Regions Visualization ===")
    
    examples = [
        ("Simple Capture", [["X","X","X"],
                           ["X","O","X"],
                           ["X","X","X"]]),
        ("Boundary Safe", [["O","X","X"],
                          ["X","O","X"],
                          ["X","X","O"]]),
        ("Mixed Case", [["X","O","X","O"],
                       ["O","X","X","O"],
                       ["X","X","O","X"],
                       ["O","X","X","X"]]),
    ]
    
    for name, board in examples:
        print(f"\n{name}:")
        
        # Original board
        print("  Original:")
        for i, row in enumerate(board):
            emoji_row = ['⬛' if cell == 'X' else '⬜' for cell in row]
            print(f"    {' '.join(emoji_row)} {row}")
        
        # Process board
        solution = Solution()
        result_board = [row[:] for row in board]
        solution.solve_approach1_boundary_dfs(result_board)
        
        print("  After capture:")
        for i, row in enumerate(result_board):
            emoji_row = ['⬛' if cell == 'X' else '⬜' for cell in row]
            print(f"    {' '.join(emoji_row)} {row}")
        
        # Analyze changes
        changes = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != result_board[i][j]:
                    changes.append((i, j, board[i][j], result_board[i][j]))
        
        if changes:
            print(f"  Changes: {changes}")
        else:
            print(f"  No changes (all 'O' regions are boundary-connected)")

if __name__ == "__main__":
    test_surrounded_regions()
    demonstrate_capture_process()
    visualize_surrounded_regions()

"""
Graph Theory Concepts:
1. Connected Components with Boundary Conditions
2. Region Capture and Boundary Analysis
3. Inverse Problem Solving (mark safe, then capture rest)
4. Multiple Traversal Strategies

Key Algorithm Insights:
- Inverse approach: Instead of finding surrounded regions, find safe regions
- Boundary-connected regions cannot be captured
- Two-phase algorithm: mark safe, then capture remaining
- Multiple implementation strategies with same core logic

Problem Solving Pattern:
1. Identify boundary-connected components
2. Mark these components as "safe"
3. Capture all remaining regions
4. Restore safe regions

Real-world Applications:
- Image processing (region filling, hole detection)
- Game development (flood fill, territory capture)
- Geographic analysis (enclosed regions, watersheds)
- Computer graphics (shape analysis, mesh processing)
- Circuit design (isolated component detection)

This problem demonstrates the power of inverse thinking in algorithm design -
instead of finding what to capture, find what NOT to capture!
"""
