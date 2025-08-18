"""
542. 01 Matrix
Difficulty: Easy

Problem:
Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.

The distance between two adjacent cells is 1.

Examples:
Input: mat = [[0,0,0],[0,1,0],[0,0,0]]
Output: [[0,0,0],[0,1,0],[0,0,0]]

Input: mat = [[0,0,0],[0,1,0],[1,1,1]]
Output: [[0,0,0],[0,1,0],[1,2,1]]

Constraints:
- m == mat.length
- n == mat[i].length
- 1 <= m, n <= 10^4
- 1 <= m * n <= 10^4
- mat[i][j] is either 0 or 1
- There is at least one 0 in mat
"""

from typing import List
from collections import deque

class Solution:
    def updateMatrix_approach1_multi_source_bfs(self, mat: List[List[int]]) -> List[List[int]]:
        """
        Approach 1: Multi-Source BFS (Optimal)
        
        Start BFS from all 0s simultaneously. Each level represents distance + 1.
        
        Time: O(M*N) - visit each cell at most once
        Space: O(M*N) - result matrix + queue
        """
        if not mat or not mat[0]:
            return mat
        
        m, n = len(mat), len(mat[0])
        result = [[float('inf')] * n for _ in range(m)]
        queue = deque()
        
        # Initialize: all 0s have distance 0, add them to queue
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    result[i][j] = 0
                    queue.append((i, j))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Multi-source BFS
        while queue:
            i, j = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and 
                    result[ni][nj] > result[i][j] + 1):
                    result[ni][nj] = result[i][j] + 1
                    queue.append((ni, nj))
        
        return result
    
    def updateMatrix_approach2_level_by_level_bfs(self, mat: List[List[int]]) -> List[List[int]]:
        """
        Approach 2: Level-by-Level BFS with Distance Tracking
        
        Process all cells at current distance before next distance.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not mat or not mat[0]:
            return mat
        
        m, n = len(mat), len(mat[0])
        result = [[-1] * n for _ in range(m)]
        queue = deque()
        
        # Initialize with all 0s
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    result[i][j] = 0
                    queue.append((i, j))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        distance = 0
        
        while queue:
            distance += 1
            size = len(queue)
            
            # Process all cells at current distance
            for _ in range(size):
                i, j = queue.popleft()
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and result[ni][nj] == -1):
                        result[ni][nj] = distance
                        queue.append((ni, nj))
        
        return result
    
    def updateMatrix_approach3_dp_two_pass(self, mat: List[List[int]]) -> List[List[int]]:
        """
        Approach 3: Dynamic Programming Two-Pass (Alternative)
        
        Use DP with two passes: top-left to bottom-right, then reverse.
        
        Time: O(M*N)
        Space: O(1) - if modifying input, O(M*N) for separate result
        """
        if not mat or not mat[0]:
            return mat
        
        m, n = len(mat), len(mat[0])
        result = [[float('inf')] * n for _ in range(m)]
        
        # Initialize 0s
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    result[i][j] = 0
        
        # First pass: top-left to bottom-right
        for i in range(m):
            for j in range(n):
                if result[i][j] != 0:
                    if i > 0:
                        result[i][j] = min(result[i][j], result[i-1][j] + 1)
                    if j > 0:
                        result[i][j] = min(result[i][j], result[i][j-1] + 1)
        
        # Second pass: bottom-right to top-left
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if result[i][j] != 0:
                    if i < m-1:
                        result[i][j] = min(result[i][j], result[i+1][j] + 1)
                    if j < n-1:
                        result[i][j] = min(result[i][j], result[i][j+1] + 1)
        
        return result
    
    def updateMatrix_approach4_bfs_with_visited(self, mat: List[List[int]]) -> List[List[int]]:
        """
        Approach 4: BFS with Explicit Visited Array
        
        Use separate visited array to track processed cells.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not mat or not mat[0]:
            return mat
        
        m, n = len(mat), len(mat[0])
        result = [[0] * n for _ in range(m)]
        visited = [[False] * n for _ in range(m)]
        queue = deque()
        
        # Initialize with all 0s
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    queue.append((i, j, 0))
                    visited[i][j] = True
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # BFS
        while queue:
            i, j, dist = queue.popleft()
            result[i][j] = dist
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and not visited[ni][nj]):
                    visited[ni][nj] = True
                    queue.append((ni, nj, dist + 1))
        
        return result

def test_01_matrix():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (mat, expected)
        ([[0,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]]),
        ([[0,0,0],[0,1,0],[1,1,1]], [[0,0,0],[0,1,0],[1,2,1]]),
        ([[0]], [[0]]),
        ([[1]], [[1]]),  # This should not happen per constraints
        ([[0,1,0],[1,1,1],[0,1,0]], [[0,1,0],[1,2,1],[0,1,0]]),
        ([[1,1,1],[1,1,1],[1,1,0]], [[4,3,2],[3,2,1],[2,1,0]]),
    ]
    
    approaches = [
        ("Multi-Source BFS", solution.updateMatrix_approach1_multi_source_bfs),
        ("Level-by-Level BFS", solution.updateMatrix_approach2_level_by_level_bfs),
        ("DP Two-Pass", solution.updateMatrix_approach3_dp_two_pass),
        ("BFS with Visited", solution.updateMatrix_approach4_bfs_with_visited),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (mat, expected) in enumerate(test_cases):
            # Skip invalid test case
            if mat == [[1]]:
                print(f"Test {i+1}: SKIP (violates constraints)")
                continue
            
            result = func(mat)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status}")
            if result != expected:
                print(f"         Input: {mat}")
                print(f"         Expected: {expected}")
                print(f"         Got: {result}")

def demonstrate_distance_calculation():
    """Demonstrate distance calculation process"""
    print("\n=== Distance Calculation Demo ===")
    
    mat = [[0,0,0],
           [0,1,0],
           [1,1,1]]
    
    print("Input matrix:")
    print_matrix(mat, "Original")
    
    # Step-by-step BFS simulation
    m, n = len(mat), len(mat[0])
    result = [[float('inf')] * n for _ in range(m)]
    queue = deque()
    
    # Initialize
    print("\nInitialization - finding all 0s:")
    for i in range(m):
        for j in range(n):
            if mat[i][j] == 0:
                result[i][j] = 0
                queue.append((i, j))
                print(f"  0 found at ({i},{j})")
    
    print_matrix(result, "After initialization")
    
    # BFS expansion
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    step = 0
    
    while queue:
        step += 1
        print(f"\nStep {step}:")
        size = len(queue)
        updates = []
        
        for _ in range(size):
            i, j = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and 
                    result[ni][nj] > result[i][j] + 1):
                    old_val = result[ni][nj]
                    result[ni][nj] = result[i][j] + 1
                    queue.append((ni, nj))
                    updates.append(f"({ni},{nj}): {old_val} -> {result[ni][nj]}")
        
        if updates:
            print(f"  Updates: {updates}")
            print_matrix(result, f"After step {step}")
        else:
            print("  No updates")
            break
    
    print(f"\nFinal distance matrix:")
    print_matrix(result, "Final result")

def print_matrix(matrix, title):
    """Helper function to print matrix"""
    print(f"  {title}:")
    for row in matrix:
        formatted_row = []
        for val in row:
            if val == float('inf'):
                formatted_row.append("∞")
            else:
                formatted_row.append(str(val))
        print(f"    {formatted_row}")

def compare_bfs_vs_dp():
    """Compare BFS and DP approaches"""
    print("\n=== BFS vs DP Comparison ===")
    
    print("Multi-Source BFS Approach:")
    print("  ✅ Intuitive: Natural distance propagation")
    print("  ✅ Single pass: O(M*N) time, each cell visited once")
    print("  ✅ Optimal: Guarantees shortest distances")
    print("  ❌ Space: Requires queue for BFS")
    
    print("\nDynamic Programming Approach:")
    print("  ✅ Space efficient: O(1) extra space possible")
    print("  ✅ No queue needed: Direct array updates")
    print("  ✅ Cache friendly: Sequential memory access")
    print("  ❌ Two passes required: Forward and backward")
    
    print("\nWhen to use each:")
    print("  • BFS: When you need to understand the propagation process")
    print("  • BFS: When working with general graphs (not just grids)")
    print("  • DP: When memory is extremely limited")
    print("  • DP: When you prefer iterative over queue-based solutions")
    
    print("\nReal-world performance:")
    print("  • Both have O(M*N) time complexity")
    print("  • BFS may have better cache locality in practice")
    print("  • DP uses less memory but requires careful implementation")

if __name__ == "__main__":
    test_01_matrix()
    demonstrate_distance_calculation()
    compare_bfs_vs_dp()

"""
Graph Theory Concepts:
1. Multi-Source Shortest Path in Grid
2. Distance Propagation from Multiple Sources
3. BFS vs DP for Distance Calculation
4. Optimal Substructure in Grid Problems

Key Algorithm Insights:
- Multi-source BFS: Start from all 0s simultaneously
- Level-order processing ensures optimal distances
- Each cell visited at most once guarantees O(M*N) time
- First visit to cell provides shortest distance

BFS vs DP Trade-offs:
- BFS: More intuitive, requires queue storage
- DP: Space efficient, requires two passes for correctness
- Both achieve optimal O(M*N) time complexity

Real-world Applications:
- Image processing (distance transforms)
- Game AI (distance fields for pathfinding)
- Computer graphics (signed distance fields)
- GIS systems (proximity analysis)
- Network analysis (multi-source shortest paths)
- Urban planning (accessibility mapping)

This problem demonstrates the elegance of multi-source BFS
for distance calculation in grid-based problems.
"""
