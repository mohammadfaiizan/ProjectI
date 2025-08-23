"""
547. Number of Provinces
Difficulty: Medium

Problem:
There are n cities. Some of them are connected, while some are not. If city a is 
connected directly with city b, and city b is connected directly with city c, then 
city a is connected indirectly with city c.

A province is a group of directly or indirectly connected cities and no other cities 
outside of the group.

You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city 
and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

Return the total number of provinces.

Examples:
Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2

Input: isConnected = [[1,0,0],[0,1,0],[0,0,1]]
Output: 3

Constraints:
- 1 <= n <= 200
- n == isConnected.length
- n == isConnected[i].length
- isConnected[i][j] is 1 or 0
- isConnected[i][i] == 1
- isConnected[i][j] == isConnected[j][i]
"""

from typing import List

class UnionFind:
    """Union-Find with path compression and union by rank"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def get_components_count(self):
        """Get number of connected components"""
        return self.components

class Solution:
    def findCircleNum_approach1_union_find(self, isConnected: List[List[int]]) -> int:
        """
        Approach 1: Union-Find (Optimal)
        
        Use Union-Find to merge connected cities and count components.
        
        Time: O(N^2 * α(N)) ≈ O(N^2)
        Space: O(N)
        """
        n = len(isConnected)
        uf = UnionFind(n)
        
        # Process all connections
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle
                if isConnected[i][j] == 1:
                    uf.union(i, j)
        
        return uf.get_components_count()
    
    def findCircleNum_approach2_dfs(self, isConnected: List[List[int]]) -> int:
        """
        Approach 2: DFS
        
        Use DFS to explore connected components.
        
        Time: O(N^2)
        Space: O(N)
        """
        n = len(isConnected)
        visited = [False] * n
        provinces = 0
        
        def dfs(city):
            """DFS to mark all cities in current province"""
            visited[city] = True
            
            for neighbor in range(n):
                if isConnected[city][neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor)
        
        for city in range(n):
            if not visited[city]:
                dfs(city)
                provinces += 1
        
        return provinces
    
    def findCircleNum_approach3_bfs(self, isConnected: List[List[int]]) -> int:
        """
        Approach 3: BFS
        
        Use BFS to explore connected components.
        
        Time: O(N^2)
        Space: O(N)
        """
        from collections import deque
        
        n = len(isConnected)
        visited = [False] * n
        provinces = 0
        
        for city in range(n):
            if not visited[city]:
                # BFS for current province
                queue = deque([city])
                visited[city] = True
                
                while queue:
                    current = queue.popleft()
                    
                    for neighbor in range(n):
                        if isConnected[current][neighbor] == 1 and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                provinces += 1
        
        return provinces
    
    def findCircleNum_approach4_iterative_dfs(self, isConnected: List[List[int]]) -> int:
        """
        Approach 4: Iterative DFS
        
        Use stack-based DFS to avoid recursion.
        
        Time: O(N^2)
        Space: O(N)
        """
        n = len(isConnected)
        visited = [False] * n
        provinces = 0
        
        for city in range(n):
            if not visited[city]:
                # Iterative DFS
                stack = [city]
                
                while stack:
                    current = stack.pop()
                    
                    if not visited[current]:
                        visited[current] = True
                        
                        # Add unvisited neighbors
                        for neighbor in range(n):
                            if isConnected[current][neighbor] == 1 and not visited[neighbor]:
                                stack.append(neighbor)
                
                provinces += 1
        
        return provinces
    
    def findCircleNum_approach5_optimized_union_find(self, isConnected: List[List[int]]) -> int:
        """
        Approach 5: Optimized Union-Find with Early Termination
        
        Add optimizations for better performance.
        
        Time: O(N^2)
        Space: O(N)
        """
        n = len(isConnected)
        parent = list(range(n))
        
        def find(x):
            """Find with path compression"""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            """Simple union"""
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y
                return True
            return False
        
        components = n
        
        # Only check upper triangle (symmetric matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j] == 1:
                    if union(i, j):
                        components -= 1
        
        return components

def test_number_of_provinces():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (isConnected, expected)
        ([[1,1,0],[1,1,0],[0,0,1]], 2),
        ([[1,0,0],[0,1,0],[0,0,1]], 3),
        ([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]], 2),
        ([[1]], 1),
        ([[1,1],[1,1]], 1),
        ([[1,0,0,0,0],[0,1,1,1,0],[0,1,1,0,0],[0,1,0,1,0],[0,0,0,0,1]], 2),
    ]
    
    approaches = [
        ("Union-Find", solution.findCircleNum_approach1_union_find),
        ("DFS", solution.findCircleNum_approach2_dfs),
        ("BFS", solution.findCircleNum_approach3_bfs),
        ("Iterative DFS", solution.findCircleNum_approach4_iterative_dfs),
        ("Optimized UF", solution.findCircleNum_approach5_optimized_union_find),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (isConnected, expected) in enumerate(test_cases):
            result = func(isConnected)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_union_find_process():
    """Demonstrate Union-Find process for province counting"""
    print("\n=== Union-Find Process Demo ===")
    
    isConnected = [[1,1,0,0],
                   [1,1,1,0],
                   [0,1,1,0],
                   [0,0,0,1]]
    
    print("Adjacency Matrix:")
    for i, row in enumerate(isConnected):
        print(f"  City {i}: {row}")
    
    print(f"\nProcessing connections:")
    
    n = len(isConnected)
    uf = UnionFind(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            if isConnected[i][j] == 1:
                print(f"  Connection found: City {i} ↔ City {j}")
                
                # Check if already connected
                if uf.find(i) == uf.find(j):
                    print(f"    Already in same province (redundant)")
                else:
                    print(f"    Union cities {i} and {j}")
                    uf.union(i, j)
                    print(f"    Components remaining: {uf.get_components_count()}")
                
                # Show current provinces
                provinces = {}
                for city in range(n):
                    root = uf.find(city)
                    if root not in provinces:
                        provinces[root] = []
                    provinces[root].append(city)
                
                province_list = list(provinces.values())
                print(f"    Current provinces: {province_list}")
    
    print(f"\nFinal result: {uf.get_components_count()} provinces")

def demonstrate_dfs_process():
    """Demonstrate DFS process for province counting"""
    print("\n=== DFS Process Demo ===")
    
    isConnected = [[1,1,0],
                   [1,1,0],
                   [0,0,1]]
    
    print("Adjacency Matrix:")
    for i, row in enumerate(isConnected):
        print(f"  City {i}: {row}")
    
    n = len(isConnected)
    visited = [False] * n
    provinces = 0
    
    def dfs_demo(city, province_id):
        print(f"    Visiting city {city} (Province {province_id})")
        visited[city] = True
        
        # Find connected neighbors
        neighbors = []
        for neighbor in range(n):
            if isConnected[city][neighbor] == 1 and not visited[neighbor]:
                neighbors.append(neighbor)
        
        if neighbors:
            print(f"    Found unvisited neighbors: {neighbors}")
            for neighbor in neighbors:
                dfs_demo(neighbor, province_id)
        else:
            print(f"    No unvisited neighbors from city {city}")
    
    print(f"\nDFS traversal:")
    
    for city in range(n):
        if not visited[city]:
            provinces += 1
            print(f"\n  Starting new province {provinces} from city {city}:")
            dfs_demo(city, provinces)
    
    print(f"\nTotal provinces found: {provinces}")

def analyze_connectivity_representations():
    """Analyze different ways to represent connectivity"""
    print("\n=== Connectivity Representations Analysis ===")
    
    print("1. **Adjacency Matrix (Given):**")
    print("   • isConnected[i][j] = 1 if cities i and j connected")
    print("   • Symmetric matrix (undirected graph)")
    print("   • Diagonal elements = 1 (city connected to itself)")
    print("   • Space: O(N²), good for dense graphs")
    
    print("\n2. **Adjacency List Alternative:**")
    print("   • List of neighbors for each city")
    print("   • More space efficient for sparse graphs")
    print("   • Easier to iterate neighbors")
    print("   • Space: O(V + E)")
    
    print("\n3. **Edge List Alternative:**")
    print("   • List of (u, v) pairs representing connections")
    print("   • Most space efficient for sparse graphs")
    print("   • Natural input for Union-Find")
    print("   • Space: O(E)")
    
    print("\nProblem Characteristics:")
    print("• **Transitive Connectivity:** If A-B and B-C, then A-C connected")
    print("• **Equivalence Relation:** Reflexive, symmetric, transitive")
    print("• **Connected Components:** Maximal connected subgraphs")
    print("• **Province = Connected Component**")
    
    print("\nAlgorithm Comparison:")
    print("• **Union-Find:** O(N² α(N)), excellent for dynamic connectivity")
    print("• **DFS/BFS:** O(N²), intuitive graph traversal")
    print("• **Tarjan's Algorithm:** O(N²), overkill for this problem")
    print("• **Matrix Multiplication:** O(N³), unnecessary complexity")

def compare_approaches_detailed():
    """Detailed comparison of different approaches"""
    print("\n=== Detailed Approach Comparison ===")
    
    print("1. **Union-Find Approach:**")
    print("   ✅ Optimal for dynamic connectivity")
    print("   ✅ Natural fit for equivalence relations")
    print("   ✅ Path compression gives near-constant operations")
    print("   ✅ Easy to implement and understand")
    print("   ❌ Requires Union-Find data structure knowledge")
    
    print("\n2. **DFS Approach:**")
    print("   ✅ Intuitive graph traversal")
    print("   ✅ Easy to understand and implement")
    print("   ✅ Standard connected components algorithm")
    print("   ✅ Can easily track component members")
    print("   ❌ Recursion depth could be issue for large graphs")
    
    print("\n3. **BFS Approach:**")
    print("   ✅ Level-by-level exploration")
    print("   ✅ Iterative (no recursion stack issues)")
    print("   ✅ Can find shortest paths within components")
    print("   ✅ Clear queue-based implementation")
    print("   ❌ Extra space for queue")
    
    print("\n4. **Iterative DFS:**")
    print("   ✅ Avoids recursion stack overflow")
    print("   ✅ Depth-first exploration pattern")
    print("   ✅ Stack-based control")
    print("   ❌ Slightly more complex than recursive DFS")
    
    print("\nWhen to Use Each:")
    print("• **Union-Find:** Dynamic connectivity, online queries")
    print("• **DFS:** Need component structure, tree-like exploration")
    print("• **BFS:** Level-by-level processing, shortest paths")
    print("• **Iterative DFS:** Large graphs, stack overflow concerns")
    
    print("\nReal-world Applications:")
    print("• **Social Networks:** Friend groups, communities")
    print("• **Computer Networks:** Network segments, subnets")
    print("• **Biological Networks:** Protein interaction groups")
    print("• **Transportation:** Connected road networks")
    print("• **Game Development:** Island/region detection")

if __name__ == "__main__":
    test_number_of_provinces()
    demonstrate_union_find_process()
    demonstrate_dfs_process()
    analyze_connectivity_representations()
    compare_approaches_detailed()

"""
Union-Find Concepts:
1. Connected Components in Undirected Graphs
2. Equivalence Relation Modeling
3. Dynamic Connectivity Queries
4. Transitive Closure of Relations

Key Problem Insights:
- Province = Connected Component
- Adjacency matrix represents undirected graph
- Transitive connectivity: A-B, B-C → A-C connected
- Count connected components efficiently

Algorithm Strategy:
1. Model cities as graph nodes
2. Process adjacency matrix connections
3. Use Union-Find to merge connected cities
4. Count remaining components as provinces

Union-Find Advantages:
- Near-constant time operations with optimizations
- Natural fit for equivalence relations
- Handles dynamic updates efficiently
- Simple implementation for counting components

Real-world Applications:
- Social network community detection
- Computer network segmentation
- Biological pathway analysis
- Transportation network design
- Geographical region analysis

This problem demonstrates Union-Find for
connected component counting in static graphs.
"""
