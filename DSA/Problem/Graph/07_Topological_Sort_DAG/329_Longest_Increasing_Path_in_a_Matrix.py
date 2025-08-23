"""
329. Longest Increasing Path in a Matrix
Difficulty: Hard

Problem:
Given an m x n integers matrix, return the length of the longest increasing path.

From each cell, you can either move in four directions: left, right, up, or down. 
You may not move diagonally or move outside of the boundary (i.e., wrap-around is not allowed).

Examples:
Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
Output: 4
Explanation: The longest increasing path is [1, 2, 6, 9].

Input: matrix = [[3,4,5],[3,2,6],[2,2,1]]
Output: 4
Explanation: The longest increasing path is [3, 4, 5, 6].

Input: matrix = [[1]]
Output: 1

Constraints:
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 200
- 0 <= matrix[i][j] <= 2^31 - 1
"""

from typing import List
from collections import deque

class Solution:
    def longestIncreasingPath_approach1_dfs_memoization(self, matrix: List[List[int]]) -> int:
        """
        Approach 1: DFS with Memoization (Optimal)
        
        Use DFS to explore all paths with memoization for efficiency.
        
        Time: O(M * N)
        Space: O(M * N)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        memo = {}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def dfs(i, j):
            """DFS to find longest increasing path starting from (i,j)"""
            if (i, j) in memo:
                return memo[(i, j)]
            
            max_length = 1  # Current cell contributes 1 to path length
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and 
                    matrix[ni][nj] > matrix[i][j]):
                    
                    max_length = max(max_length, 1 + dfs(ni, nj))
            
            memo[(i, j)] = max_length
            return max_length
        
        # Try starting from each cell
        result = 0
        for i in range(m):
            for j in range(n):
                result = max(result, dfs(i, j))
        
        return result
    
    def longestIncreasingPath_approach2_topological_sort(self, matrix: List[List[int]]) -> int:
        """
        Approach 2: Topological Sort on Implicit DAG
        
        Treat matrix as DAG where edges go from smaller to larger values.
        
        Time: O(M * N)
        Space: O(M * N)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Calculate in-degrees for each cell
        in_degree = [[0] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < m and 0 <= nj < n and 
                        matrix[ni][nj] > matrix[i][j]):
                        in_degree[ni][nj] += 1
        
        # Initialize queue with cells having in-degree 0
        queue = deque()
        for i in range(m):
            for j in range(n):
                if in_degree[i][j] == 0:
                    queue.append((i, j))
        
        # Process cells level by level
        max_length = 0
        
        while queue:
            level_size = len(queue)
            max_length += 1
            
            for _ in range(level_size):
                i, j = queue.popleft()
                
                # Update neighbors
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        matrix[ni][nj] > matrix[i][j]):
                        
                        in_degree[ni][nj] -= 1
                        if in_degree[ni][nj] == 0:
                            queue.append((ni, nj))
        
        return max_length
    
    def longestIncreasingPath_approach3_dp_bottom_up(self, matrix: List[List[int]]) -> int:
        """
        Approach 3: Dynamic Programming Bottom-Up
        
        Process cells in sorted order of values.
        
        Time: O(M * N * log(M * N))
        Space: O(M * N)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Create list of cells sorted by value
        cells = []
        for i in range(m):
            for j in range(n):
                cells.append((matrix[i][j], i, j))
        
        cells.sort()
        
        # DP array to store longest path ending at each cell
        dp = [[1] * n for _ in range(m)]
        
        # Process cells in increasing order of values
        for value, i, j in cells:
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and 
                    matrix[ni][nj] < matrix[i][j]):
                    
                    dp[i][j] = max(dp[i][j], dp[ni][nj] + 1)
        
        # Find maximum path length
        result = 0
        for i in range(m):
            for j in range(n):
                result = max(result, dp[i][j])
        
        return result
    
    def longestIncreasingPath_approach4_iterative_dfs(self, matrix: List[List[int]]) -> int:
        """
        Approach 4: Iterative DFS with Stack
        
        Avoid recursion using explicit stack for DFS.
        
        Time: O(M * N)
        Space: O(M * N)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        memo = {}
        
        def iterative_dfs(start_i, start_j):
            """Iterative DFS using explicit stack"""
            if (start_i, start_j) in memo:
                return memo[(start_i, start_j)]
            
            stack = [(start_i, start_j, False)]  # (i, j, processed)
            path_length = {}
            
            while stack:
                i, j, processed = stack.pop()
                
                if processed:
                    # Post-processing: calculate path length
                    max_length = 1
                    
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        
                        if (0 <= ni < m and 0 <= nj < n and 
                            matrix[ni][nj] > matrix[i][j]):
                            
                            if (ni, nj) in path_length:
                                max_length = max(max_length, 1 + path_length[(ni, nj)])
                    
                    path_length[(i, j)] = max_length
                    memo[(i, j)] = max_length
                else:
                    if (i, j) in memo:
                        path_length[(i, j)] = memo[(i, j)]
                        continue
                    
                    # Pre-processing: add neighbors to stack
                    stack.append((i, j, True))
                    
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        
                        if (0 <= ni < m and 0 <= nj < n and 
                            matrix[ni][nj] > matrix[i][j] and 
                            (ni, nj) not in memo):
                            
                            stack.append((ni, nj, False))
            
            return memo.get((start_i, start_j), 1)
        
        # Try starting from each cell
        result = 0
        for i in range(m):
            for j in range(n):
                result = max(result, iterative_dfs(i, j))
        
        return result
    
    def longestIncreasingPath_approach5_peeling_algorithm(self, matrix: List[List[int]]) -> int:
        """
        Approach 5: Peeling Algorithm (Onion Layer Approach)
        
        Remove boundary cells layer by layer like peeling an onion.
        
        Time: O(M * N)
        Space: O(M * N)
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Calculate out-degrees (number of increasing neighbors)
        out_degree = [[0] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < m and 0 <= nj < n and 
                        matrix[ni][nj] > matrix[i][j]):
                        out_degree[i][j] += 1
        
        # Initialize queue with leaves (out-degree 0)
        queue = deque()
        for i in range(m):
            for j in range(n):
                if out_degree[i][j] == 0:
                    queue.append((i, j))
        
        # Peel layers
        height = 0
        
        while queue:
            level_size = len(queue)
            height += 1
            
            for _ in range(level_size):
                i, j = queue.popleft()
                
                # Update predecessors (cells that can reach current cell)
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        matrix[ni][nj] < matrix[i][j]):
                        
                        out_degree[ni][nj] -= 1
                        if out_degree[ni][nj] == 0:
                            queue.append((ni, nj))
        
        return height

def test_longest_increasing_path():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (matrix, expected)
        ([[9,9,4],[6,6,8],[2,1,1]], 4),
        ([[3,4,5],[3,2,6],[2,2,1]], 4),
        ([[1]], 1),
        ([[1,2],[3,4]], 3),
        ([[7,8,9],[9,7,6],[7,2,3]], 6),
        ([[0,1,2,3,4,5,6,7,8,9]], 10),
    ]
    
    approaches = [
        ("DFS Memoization", solution.longestIncreasingPath_approach1_dfs_memoization),
        ("Topological Sort", solution.longestIncreasingPath_approach2_topological_sort),
        ("DP Bottom-Up", solution.longestIncreasingPath_approach3_dp_bottom_up),
        ("Iterative DFS", solution.longestIncreasingPath_approach4_iterative_dfs),
        ("Peeling Algorithm", solution.longestIncreasingPath_approach5_peeling_algorithm),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (matrix, expected) in enumerate(test_cases):
            result = func([row[:] for row in matrix])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_path_exploration():
    """Demonstrate longest increasing path exploration"""
    print("\n=== Path Exploration Demo ===")
    
    matrix = [[9,9,4],[6,6,8],[2,1,1]]
    
    print(f"Matrix:")
    for i, row in enumerate(matrix):
        print(f"  {i}: {row}")
    
    print(f"\nFinding longest increasing paths:")
    
    # Manually trace some paths
    paths = [
        {"start": (2,1), "path": [(2,1), (2,0), (1,0), (0,0)], "values": [1,2,6,9], "length": 4},
        {"start": (2,2), "path": [(2,2), (1,2), (0,2)], "values": [1,8,4], "length": 2},
        {"start": (0,2), "path": [(0,2), (1,2)], "values": [4,8], "length": 2},
    ]
    
    for i, path_info in enumerate(paths, 1):
        print(f"\nPath {i} starting from {path_info['start']}:")
        print(f"  Route: {path_info['path']}")
        print(f"  Values: {path_info['values']}")
        print(f"  Length: {path_info['length']}")
        print(f"  Increasing: {all(path_info['values'][i] < path_info['values'][i+1] for i in range(len(path_info['values'])-1))}")
    
    # Find actual longest path
    solution = Solution()
    result = solution.longestIncreasingPath_approach1_dfs_memoization(matrix)
    print(f"\nLongest increasing path length: {result}")

def demonstrate_topological_sort_on_matrix():
    """Demonstrate topological sort approach on matrix"""
    print("\n=== Topological Sort on Matrix Demo ===")
    
    matrix = [[3,4,5],[3,2,6],[2,2,1]]
    m, n = len(matrix), len(matrix[0])
    
    print(f"Matrix:")
    for i, row in enumerate(matrix):
        print(f"  {i}: {row}")
    
    # Calculate in-degrees
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    in_degree = [[0] * n for _ in range(m)]
    
    print(f"\nCalculating in-degrees (number of smaller neighbors):")
    
    for i in range(m):
        for j in range(n):
            neighbors = []
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if (0 <= ni < m and 0 <= nj < n):
                    neighbors.append((ni, nj, matrix[ni][nj]))
                    if matrix[ni][nj] < matrix[i][j]:
                        in_degree[i][j] += 1
            
            print(f"  ({i},{j}) value={matrix[i][j]}: neighbors={neighbors}, in_degree={in_degree[i][j]}")
    
    print(f"\nIn-degree matrix:")
    for row in in_degree:
        print(f"  {row}")
    
    # Find cells with in-degree 0 (local minima)
    queue = deque()
    for i in range(m):
        for j in range(n):
            if in_degree[i][j] == 0:
                queue.append((i, j))
    
    print(f"\nInitial queue (local minima): {list(queue)}")
    
    # Process level by level
    level = 0
    while queue:
        level += 1
        level_size = len(queue)
        level_cells = []
        
        print(f"\nLevel {level}: Processing {level_size} cells")
        
        for _ in range(level_size):
            i, j = queue.popleft()
            level_cells.append((i, j, matrix[i][j]))
            
            # Update neighbors
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and 
                    matrix[ni][nj] > matrix[i][j]):
                    
                    in_degree[ni][nj] -= 1
                    if in_degree[ni][nj] == 0:
                        queue.append((ni, nj))
        
        print(f"  Cells: {level_cells}")
    
    print(f"\nLongest increasing path length: {level}")

def analyze_algorithm_comparison():
    """Analyze comparison of different algorithms"""
    print("\n=== Algorithm Comparison Analysis ===")
    
    print("Longest Increasing Path Algorithm Comparison:")
    
    print("\n1. **DFS with Memoization:**")
    print("   • Time: O(M * N)")
    print("   • Space: O(M * N)")
    print("   • Natural recursive approach")
    print("   • Memoization prevents recomputation")
    print("   • Most intuitive implementation")
    
    print("\n2. **Topological Sort:**")
    print("   • Time: O(M * N)")
    print("   • Space: O(M * N)")
    print("   • Treats matrix as implicit DAG")
    print("   • Processes cells level by level")
    print("   • Good for understanding structure")
    
    print("\n3. **Dynamic Programming:**")
    print("   • Time: O(M * N * log(M * N))")
    print("   • Space: O(M * N)")
    print("   • Processes cells in sorted order")
    print("   • Bottom-up approach")
    print("   • Clear dependency resolution")
    
    print("\n4. **Peeling Algorithm:**")
    print("   • Time: O(M * N)")
    print("   • Space: O(M * N)")
    print("   • Removes boundary cells iteratively")
    print("   • Similar to topological sort")
    print("   • Intuitive layer-by-layer processing")
    
    print("\nKey Insights:")
    print("• All optimal algorithms have O(M * N) time complexity")
    print("• Problem can be viewed as DAG longest path")
    print("• Memoization essential for avoiding exponential time")
    print("• Multiple valid approaches with same efficiency")

def demonstrate_practical_applications():
    """Demonstrate practical applications of longest increasing path"""
    print("\n=== Practical Applications ===")
    
    print("Longest Increasing Path Applications:")
    
    print("\n1. **Terrain Analysis:**")
    print("   • Find longest ascending hiking trail")
    print("   • Water flow path analysis")
    print("   • Elevation profile optimization")
    print("   • Geographical feature mapping")
    
    print("\n2. **Game Development:**")
    print("   • Character progression paths")
    print("   • Level difficulty curves")
    print("   • Resource gathering optimization")
    print("   • Skill tree advancement")
    
    print("\n3. **Financial Analysis:**")
    print("   • Stock price trend analysis")
    print("   • Portfolio growth paths")
    print("   • Investment strategy optimization")
    print("   • Risk-return trajectory analysis")
    
    print("\n4. **Image Processing:**")
    print("   • Edge detection and following")
    print("   • Feature extraction")
    print("   • Pattern recognition")
    print("   • Image segmentation")
    
    print("\n5. **Network Analysis:**")
    print("   • Social influence propagation")
    print("   • Information cascade analysis")
    print("   • Viral spread modeling")
    print("   • Network growth patterns")
    
    print("\nAlgorithmic Insights:")
    print("• DAG structure enables efficient DP solutions")
    print("• Memoization crucial for avoiding redundant computation")
    print("• Multiple algorithmic perspectives provide different insights")
    print("• Topological ordering reveals natural processing sequence")

if __name__ == "__main__":
    test_longest_increasing_path()
    demonstrate_path_exploration()
    demonstrate_topological_sort_on_matrix()
    analyze_algorithm_comparison()
    demonstrate_practical_applications()

"""
Longest Increasing Path and DAG DP Concepts:
1. DFS with Memoization for Path Optimization
2. Topological Sort on Implicit Matrix DAG
3. Dynamic Programming on Directed Acyclic Graphs
4. Matrix as Graph Representation
5. Layer-by-Layer Processing Algorithms

Key Problem Insights:
- Matrix cells as vertices, increasing edges as directed graph
- Longest path in DAG solvable with DP in O(V + E) time
- Memoization prevents exponential time complexity
- Multiple algorithmic perspectives yield same optimal result

Algorithm Strategy:
1. DFS + Memoization: Explore paths recursively with caching
2. Topological Sort: Process cells in dependency order
3. Both approaches: O(M * N) time and space complexity
4. Matrix treated as implicit directed acyclic graph

DAG Structure in Matrix:
- Directed edges from smaller to larger values
- No cycles possible (values must strictly increase)
- Topological ordering exists and is computable
- Longest path solvable efficiently with DP

Memoization Optimization:
- Cache results for each cell to avoid recomputation
- Reduces time complexity from exponential to linear
- Essential for practical implementation
- Space-time trade-off for significant performance gain

Matrix as Graph Applications:
- Terrain analysis and elevation modeling
- Game development and progression systems
- Financial trend analysis and optimization
- Image processing and pattern recognition
- Network analysis and influence propagation

This problem demonstrates advanced DP techniques
on implicit DAG structures with practical applications.
"""
