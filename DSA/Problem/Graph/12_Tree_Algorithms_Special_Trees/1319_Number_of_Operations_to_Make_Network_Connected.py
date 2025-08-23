"""
1319. Number of Operations to Make Network Connected - Multiple Approaches
Difficulty: Medium

There are n computers numbered from 0 to n - 1 connected by ethernet cables connections forming a network where connections[i] = [ai, bi] represents a connection between computers ai and bi. Any computer can reach any other computer directly or indirectly through the network.

You are given an initial computer network connections. You can extract certain cables between two directly connected computers, and place them between any pair of disconnected computers to make them directly connected.

Return the minimum number of operations needed to make all the computers connected. If it is not possible, return -1.
"""

from typing import List, Set
from collections import defaultdict, deque

class NetworkConnected:
    """Multiple approaches to find minimum operations to connect network"""
    
    def makeConnected_union_find(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 1: Union-Find (Disjoint Set Union)
        
        Use Union-Find to count connected components and redundant edges.
        
        Time: O(E * α(V)), Space: O(V)
        """
        if len(connections) < n - 1:
            return -1  # Not enough cables to connect all computers
        
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
                self.components = n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False  # Already connected (redundant edge)
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                
                self.components -= 1
                return True
        
        uf = UnionFind(n)
        redundant_cables = 0
        
        for a, b in connections:
            if not uf.union(a, b):
                redundant_cables += 1
        
        # Need (components - 1) cables to connect all components
        needed_cables = uf.components - 1
        
        return needed_cables if redundant_cables >= needed_cables else -1
    
    def makeConnected_dfs(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 2: DFS to Count Connected Components
        
        Use DFS to find connected components and count redundant edges.
        
        Time: O(V + E), Space: O(V + E)
        """
        if len(connections) < n - 1:
            return -1
        
        # Build adjacency list
        graph = defaultdict(list)
        edge_set = set()
        
        for a, b in connections:
            graph[a].append(b)
            graph[b].append(a)
            # Count unique edges
            edge_set.add((min(a, b), max(a, b)))
        
        # Count connected components using DFS
        visited = [False] * n
        components = 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1
        
        # Calculate redundant cables
        unique_edges = len(edge_set)
        min_edges_needed = n - 1
        redundant_cables = unique_edges - (n - components)
        
        return components - 1
    
    def makeConnected_bfs(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 3: BFS to Count Connected Components
        
        Use BFS to find connected components.
        
        Time: O(V + E), Space: O(V + E)
        """
        if len(connections) < n - 1:
            return -1
        
        # Build adjacency list
        graph = defaultdict(list)
        for a, b in connections:
            graph[a].append(b)
            graph[b].append(a)
        
        visited = [False] * n
        components = 0
        
        for i in range(n):
            if not visited[i]:
                # BFS from unvisited node
                queue = deque([i])
                visited[i] = True
                
                while queue:
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                components += 1
        
        return components - 1
    
    def makeConnected_optimized_union_find(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 4: Optimized Union-Find with Path Compression and Union by Size
        
        Enhanced Union-Find with better optimization techniques.
        
        Time: O(E * α(V)), Space: O(V)
        """
        if len(connections) < n - 1:
            return -1
        
        class OptimizedUnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.size = [1] * n
                self.components = n
            
            def find(self, x):
                # Path compression
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                root_x, root_y = self.find(x), self.find(y)
                if root_x == root_y:
                    return False
                
                # Union by size
                if self.size[root_x] < self.size[root_y]:
                    root_x, root_y = root_y, root_x
                
                self.parent[root_y] = root_x
                self.size[root_x] += self.size[root_y]
                self.components -= 1
                return True
        
        uf = OptimizedUnionFind(n)
        
        for a, b in connections:
            uf.union(a, b)
        
        return uf.components - 1
    
    def makeConnected_iterative_merging(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 5: Iterative Component Merging
        
        Iteratively merge components and track the process.
        
        Time: O(V + E), Space: O(V)
        """
        if len(connections) < n - 1:
            return -1
        
        # Initialize each computer as its own component
        component_id = list(range(n))
        component_size = [1] * n
        num_components = n
        
        def find_component(x):
            # Find root component with path compression
            if component_id[x] != x:
                component_id[x] = find_component(component_id[x])
            return component_id[x]
        
        def merge_components(x, y):
            nonlocal num_components
            
            root_x = find_component(x)
            root_y = find_component(y)
            
            if root_x == root_y:
                return False  # Already in same component
            
            # Merge smaller component into larger one
            if component_size[root_x] < component_size[root_y]:
                root_x, root_y = root_y, root_x
            
            component_id[root_y] = root_x
            component_size[root_x] += component_size[root_y]
            num_components -= 1
            return True
        
        # Process all connections
        for a, b in connections:
            merge_components(a, b)
        
        return num_components - 1
    
    def makeConnected_graph_theory_approach(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 6: Pure Graph Theory Approach
        
        Use graph theory principles to solve the problem.
        
        Time: O(V + E), Space: O(V + E)
        """
        if len(connections) < n - 1:
            return -1
        
        # Build adjacency list and count edges
        adj = [[] for _ in range(n)]
        edges = set()
        
        for a, b in connections:
            adj[a].append(b)
            adj[b].append(a)
            edges.add((min(a, b), max(a, b)))
        
        # Find connected components using DFS
        visited = [False] * n
        components = []
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)
        
        # Calculate minimum operations needed
        num_components = len(components)
        total_edges = len(edges)
        
        # In a connected graph with n nodes, we need exactly n-1 edges
        # We have 'total_edges' edges and 'num_components' components
        # Each component needs to be connected, requiring (num_components - 1) additional edges
        
        return num_components - 1

def test_network_connected():
    """Test network connected algorithms"""
    solver = NetworkConnected()
    
    test_cases = [
        (4, [[0,1],[0,2],[1,2]], 1, "Triangle with isolated node"),
        (6, [[0,1],[0,2],[0,3],[1,2],[1,3]], 2, "Two components"),
        (6, [[0,1],[0,2],[0,3],[1,2]], -1, "Not enough cables"),
        (5, [[0,1],[0,2],[3,4],[2,3]], 0, "Already connected"),
        (4, [[0,1],[0,2],[1,2],[2,3]], 0, "Linear connection"),
    ]
    
    algorithms = [
        ("Union-Find", solver.makeConnected_union_find),
        ("DFS", solver.makeConnected_dfs),
        ("BFS", solver.makeConnected_bfs),
        ("Optimized Union-Find", solver.makeConnected_optimized_union_find),
        ("Iterative Merging", solver.makeConnected_iterative_merging),
        ("Graph Theory", solver.makeConnected_graph_theory_approach),
    ]
    
    print("=== Testing Number of Operations to Make Network Connected ===")
    
    for n, connections, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"n={n}, connections={connections}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, connections)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Operations: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_network_connected()

"""
Number of Operations to Make Network Connected demonstrates
Union-Find data structures, connected component analysis,
and graph connectivity optimization problems.
"""
