"""
547. Number of Provinces - Multiple Approaches
Difficulty: Medium

There are n cities. Some of them are connected, while some are not. If city a is connected directly with city b, and city b is connected directly with city c, then city a is connected indirectly with city c.

A province is a group of directly or indirectly connected cities and no other cities outside of the group.

You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

Return the total number of provinces.
"""

from typing import List, Set
from collections import deque

class NumberOfProvinces:
    """Multiple approaches to count connected components (provinces)"""
    
    def findCircleNum_dfs_recursive(self, isConnected: List[List[int]]) -> int:
        """
        Approach 1: Recursive DFS
        
        Use recursive DFS to explore each connected component.
        
        Time: O(n²), Space: O(n)
        """
        n = len(isConnected)
        visited = [False] * n
        provinces = 0
        
        def dfs(city: int):
            """DFS to visit all cities in current province"""
            visited[city] = True
            
            for neighbor in range(n):
                if (isConnected[city][neighbor] == 1 and 
                    not visited[neighbor]):
                    dfs(neighbor)
        
        # Count connected components
        for city in range(n):
            if not visited[city]:
                dfs(city)
                provinces += 1
        
        return provinces
    
    def findCircleNum_dfs_iterative(self, isConnected: List[List[int]]) -> int:
        """
        Approach 2: Iterative DFS using Stack
        
        Use explicit stack for DFS to avoid recursion depth issues.
        
        Time: O(n²), Space: O(n)
        """
        n = len(isConnected)
        visited = [False] * n
        provinces = 0
        
        for city in range(n):
            if not visited[city]:
                # Start DFS from unvisited city
                stack = [city]
                
                while stack:
                    current = stack.pop()
                    
                    if not visited[current]:
                        visited[current] = True
                        
                        # Add all unvisited neighbors to stack
                        for neighbor in range(n):
                            if (isConnected[current][neighbor] == 1 and 
                                not visited[neighbor]):
                                stack.append(neighbor)
                
                provinces += 1
        
        return provinces
    
    def findCircleNum_bfs(self, isConnected: List[List[int]]) -> int:
        """
        Approach 3: BFS using Queue
        
        Use BFS to explore each connected component level by level.
        
        Time: O(n²), Space: O(n)
        """
        n = len(isConnected)
        visited = [False] * n
        provinces = 0
        
        for city in range(n):
            if not visited[city]:
                # Start BFS from unvisited city
                queue = deque([city])
                visited[city] = True
                
                while queue:
                    current = queue.popleft()
                    
                    # Add all unvisited neighbors to queue
                    for neighbor in range(n):
                        if (isConnected[current][neighbor] == 1 and 
                            not visited[neighbor]):
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                provinces += 1
        
        return provinces
    
    def findCircleNum_union_find(self, isConnected: List[List[int]]) -> int:
        """
        Approach 4: Union-Find (Disjoint Set Union)
        
        Use Union-Find to group connected cities and count components.
        
        Time: O(n² α(n)), Space: O(n)
        """
        n = len(isConnected)
        
        # Union-Find implementation
        parent = list(range(n))
        rank = [0] * n
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x: int, y: int):
            px, py = find(x), find(y)
            
            if px != py:
                # Union by rank
                if rank[px] < rank[py]:
                    parent[px] = py
                elif rank[px] > rank[py]:
                    parent[py] = px
                else:
                    parent[py] = px
                    rank[px] += 1
        
        # Union connected cities
        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j] == 1:
                    union(i, j)
        
        # Count unique roots (provinces)
        return len(set(find(i) for i in range(n)))
    
    def findCircleNum_adjacency_list_dfs(self, isConnected: List[List[int]]) -> int:
        """
        Approach 5: Convert to Adjacency List + DFS
        
        Convert matrix to adjacency list for more efficient traversal.
        
        Time: O(n²), Space: O(n²)
        """
        n = len(isConnected)
        
        # Build adjacency list
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and isConnected[i][j] == 1:
                    graph[i].append(j)
        
        visited = [False] * n
        provinces = 0
        
        def dfs(city: int):
            visited[city] = True
            for neighbor in graph[city]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        # Count connected components
        for city in range(n):
            if not visited[city]:
                dfs(city)
                provinces += 1
        
        return provinces
    
    def findCircleNum_optimized_matrix(self, isConnected: List[List[int]]) -> int:
        """
        Approach 6: Optimized Matrix Traversal
        
        Optimize matrix traversal by marking visited cities in the matrix.
        
        Time: O(n²), Space: O(1) - modifies input
        """
        n = len(isConnected)
        provinces = 0
        
        def dfs(city: int):
            for neighbor in range(n):
                if isConnected[city][neighbor] == 1:
                    isConnected[city][neighbor] = 0  # Mark as visited
                    isConnected[neighbor][city] = 0  # Mark symmetric entry
                    dfs(neighbor)
        
        for city in range(n):
            if isConnected[city][city] == 1:  # Unvisited city
                dfs(city)
                provinces += 1
        
        return provinces
    
    def findCircleNum_bit_manipulation(self, isConnected: List[List[int]]) -> int:
        """
        Approach 7: Bit Manipulation for Visited Tracking
        
        Use bit manipulation to track visited cities efficiently.
        
        Time: O(n²), Space: O(1)
        """
        n = len(isConnected)
        visited = 0  # Bitmask for visited cities
        provinces = 0
        
        def dfs(city: int):
            visited_bit = 1 << city
            nonlocal visited
            visited |= visited_bit
            
            for neighbor in range(n):
                neighbor_bit = 1 << neighbor
                if (isConnected[city][neighbor] == 1 and 
                    (visited & neighbor_bit) == 0):
                    dfs(neighbor)
        
        for city in range(n):
            city_bit = 1 << city
            if (visited & city_bit) == 0:
                dfs(city)
                provinces += 1
        
        return provinces
    
    def findCircleNum_parallel_union_find(self, isConnected: List[List[int]]) -> int:
        """
        Approach 8: Optimized Union-Find with Path Compression
        
        Enhanced Union-Find with both path compression and union by rank.
        
        Time: O(n² α(n)), Space: O(n)
        """
        n = len(isConnected)
        
        parent = list(range(n))
        size = [1] * n  # Size of each component
        components = n  # Initial number of components
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x: int, y: int) -> bool:
            nonlocal components
            px, py = find(x), find(y)
            
            if px == py:
                return False
            
            # Union by size (larger component becomes parent)
            if size[px] < size[py]:
                px, py = py, px
            
            parent[py] = px
            size[px] += size[py]
            components -= 1
            return True
        
        # Process all connections
        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j] == 1:
                    union(i, j)
        
        return components
    
    def findCircleNum_matrix_powers(self, isConnected: List[List[int]]) -> int:
        """
        Approach 9: Matrix Powers for Transitive Closure
        
        Use matrix multiplication to find transitive closure.
        
        Time: O(n³), Space: O(n²)
        """
        n = len(isConnected)
        
        # Create adjacency matrix copy
        matrix = [row[:] for row in isConnected]
        
        # Floyd-Warshall to find transitive closure
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    matrix[i][j] = matrix[i][j] or (matrix[i][k] and matrix[k][j])
        
        # Count unique rows (each represents a province)
        visited = [False] * n
        provinces = 0
        
        for i in range(n):
            if not visited[i]:
                # Mark all cities in same province
                for j in range(n):
                    if matrix[i][j] == 1:
                        visited[j] = True
                provinces += 1
        
        return provinces

def test_number_of_provinces():
    """Test number of provinces algorithms"""
    solver = NumberOfProvinces()
    
    test_cases = [
        ([[1,1,0],[1,1,0],[0,0,1]], 2, "Two provinces"),
        ([[1,0,0],[0,1,0],[0,0,1]], 3, "Three isolated cities"),
        ([[1,1,1],[1,1,1],[1,1,1]], 1, "All connected"),
        ([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]], 2, "Two pairs"),
        ([[1]], 1, "Single city"),
    ]
    
    algorithms = [
        ("Recursive DFS", solver.findCircleNum_dfs_recursive),
        ("Iterative DFS", solver.findCircleNum_dfs_iterative),
        ("BFS", solver.findCircleNum_bfs),
        ("Union-Find", solver.findCircleNum_union_find),
        ("Adjacency List DFS", solver.findCircleNum_adjacency_list_dfs),
        ("Bit Manipulation", solver.findCircleNum_bit_manipulation),
        ("Optimized Union-Find", solver.findCircleNum_parallel_union_find),
        ("Matrix Powers", solver.findCircleNum_matrix_powers),
    ]
    
    print("=== Testing Number of Provinces ===")
    
    for isConnected, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Matrix: {isConnected}")
        
        for alg_name, alg_func in algorithms:
            try:
                # Create copy since some algorithms modify input
                matrix_copy = [row[:] for row in isConnected]
                result = alg_func(matrix_copy)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Provinces: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_number_of_provinces()

"""
Number of Provinces demonstrates fundamental connected components
algorithms using DFS, BFS, and Union-Find data structures
for graph connectivity analysis.
"""
