"""
1319. Number of Operations to Make Network Connected
Difficulty: Medium

Problem:
You are given an integer n. There are n computers numbered from 0 to n-1 connected 
by ethernet cables connections forming a network where connections[i] = [ai, bi] 
connects computers ai and bi.

Any computer can reach any other computer directly or indirectly through the network.

You are given an initial computer network connections. You can extract certain cables 
between two directly connected computers, and place them between any pair of disconnected 
computers to make them directly connected.

Return the minimum number of times you need to do this to make all the computers connected. 
If it's not possible, return -1.

Examples:
Input: n = 4, connections = [[0,1],[0,2],[1,2]]
Output: 1
Explanation: Remove cable between computer 1 and 2 and place between computers 1 and 3.

Input: n = 6, connections = [[0,1],[0,2],[0,3],[1,2],[1,3]]
Output: 2

Input: n = 6, connections = [[0,1],[0,2],[0,3],[1,2]]
Output: -1

Constraints:
- 1 <= n <= 10^5
- 1 <= connections.length <= min(n*(n-1)/2, 10^5)
- connections[i].length == 2
- 0 <= ai, bi < n
- ai != bi
- There are no repeated connections
"""

from typing import List
from collections import defaultdict

class Solution:
    def makeConnected_approach1_union_find(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 1: Union-Find with redundant edge counting
        
        Key insights:
        1. To connect n computers, we need at least n-1 cables
        2. Number of operations = number of connected components - 1
        3. Extra cables = total cables - (n - connected_components)
        
        Time: O(E * α(N)) where α is inverse Ackermann function
        Space: O(N)
        """
        # Check if we have enough cables
        if len(connections) < n - 1:
            return -1
        
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
        redundant_edges = 0
        
        # Process all connections
        for a, b in connections:
            if not uf.union(a, b):
                redundant_edges += 1
        
        # We need (components - 1) cables to connect all components
        # Each operation reduces components by 1
        return uf.components - 1
    
    def makeConnected_approach2_dfs_components(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 2: DFS to find connected components
        
        Build adjacency list and use DFS to count connected components.
        
        Time: O(N + E)
        Space: O(N + E)
        """
        # Check if we have enough cables
        if len(connections) < n - 1:
            return -1
        
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in connections:
            adj[a].append(b)
            adj[b].append(a)
        
        # Count connected components using DFS
        visited = [False] * n
        components = 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1
        
        return components - 1
    
    def makeConnected_approach3_bfs_components(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 3: BFS to find connected components
        
        Use BFS instead of DFS for component counting.
        
        Time: O(N + E)
        Space: O(N + E)
        """
        from collections import deque
        
        if len(connections) < n - 1:
            return -1
        
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in connections:
            adj[a].append(b)
            adj[b].append(a)
        
        visited = [False] * n
        components = 0
        
        for i in range(n):
            if not visited[i]:
                # BFS for current component
                queue = deque([i])
                visited[i] = True
                
                while queue:
                    node = queue.popleft()
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                components += 1
        
        return components - 1
    
    def makeConnected_approach4_mathematical_analysis(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 4: Mathematical approach with detailed analysis
        
        Analyze the problem from graph theory perspective.
        
        Time: O(E * α(N))
        Space: O(N)
        """
        total_edges = len(connections)
        
        # Minimum edges needed for connected graph
        min_edges_needed = n - 1
        
        if total_edges < min_edges_needed:
            return -1
        
        # Use Union-Find to find actual structure
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
                return True
            return False
        
        # Process connections and count redundant edges
        redundant_edges = 0
        for a, b in connections:
            if not union(a, b):
                redundant_edges += 1
        
        # Count connected components
        components = len(set(find(i) for i in range(n)))
        
        # Verify we have enough redundant edges to connect components
        operations_needed = components - 1
        
        return operations_needed
    
    def makeConnected_approach5_optimized_union_find(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 5: Optimized Union-Find with path compression and union by rank
        
        Most efficient implementation with all optimizations.
        
        Time: O(E * α(N)) ≈ O(E) for practical purposes
        Space: O(N)
        """
        if len(connections) < n - 1:
            return -1
        
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            
            return True
        
        # Process all connections
        for a, b in connections:
            union(a, b)
        
        # Count unique components
        components = len(set(find(i) for i in range(n)))
        
        return components - 1

def test_make_connected():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, connections, expected)
        (4, [[0,1],[0,2],[1,2]], 1),
        (6, [[0,1],[0,2],[0,3],[1,2],[1,3]], 2),
        (6, [[0,1],[0,2],[0,3],[1,2]], -1),
        (5, [[0,1],[0,2],[3,4]], 1),  # Two components
        (4, [[0,1]], -1),  # Not enough cables
        (1, [], 0),  # Single computer
        (3, [[0,1],[1,2]], 0),  # Already connected
        (5, [[0,1],[1,2],[2,3],[3,4],[0,2],[1,3]], 0),  # Over-connected
    ]
    
    approaches = [
        ("Union-Find", solution.makeConnected_approach1_union_find),
        ("DFS Components", solution.makeConnected_approach2_dfs_components),
        ("BFS Components", solution.makeConnected_approach3_bfs_components),
        ("Mathematical Analysis", solution.makeConnected_approach4_mathematical_analysis),
        ("Optimized Union-Find", solution.makeConnected_approach5_optimized_union_find),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, connections, expected) in enumerate(test_cases):
            result = func(n, connections)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, connections={connections}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_network_analysis():
    """Demonstrate network connectivity analysis"""
    print("\n=== Network Connectivity Analysis ===")
    
    n = 6
    connections = [[0,1],[0,2],[0,3],[1,2],[1,3]]
    
    print(f"Network: {n} computers")
    print(f"Connections: {connections}")
    
    # Analyze using Union-Find
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px
            return True
        return False
    
    redundant_edges = []
    essential_edges = []
    
    for a, b in connections:
        if union(a, b):
            essential_edges.append([a, b])
        else:
            redundant_edges.append([a, b])
    
    # Find components
    components = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)
    
    print(f"\nAnalysis:")
    print(f"Connected components: {len(components)}")
    for i, (root, members) in enumerate(components.items()):
        print(f"  Component {i+1}: {sorted(members)}")
    
    print(f"\nEdge classification:")
    print(f"Essential edges: {essential_edges}")
    print(f"Redundant edges: {redundant_edges}")
    
    print(f"\nSolution:")
    operations_needed = len(components) - 1
    redundant_available = len(redundant_edges)
    print(f"Operations needed: {operations_needed}")
    print(f"Redundant edges available: {redundant_available}")
    print(f"Possible: {redundant_available >= operations_needed}")

def analyze_minimum_spanning_tree_connection():
    """Analyze the relationship to minimum spanning trees"""
    print("\n=== MST Connection Analysis ===")
    
    print("Key insights:")
    print("1. To connect n computers, we need exactly n-1 edges")
    print("2. Any connected graph on n vertices has at least n-1 edges")
    print("3. Extra edges beyond n-1 can be relocated to connect components")
    print("4. This problem is related to finding connected components")
    
    print("\nMathematical relationships:")
    print("- Components = number of disjoint sets after Union-Find")
    print("- Operations needed = Components - 1")
    print("- Minimum edges needed = n - 1")
    print("- Success condition: total_edges >= n - 1")
    
    examples = [
        (4, 3, 2, 1),  # n=4, edges=3, components=2, operations=1
        (6, 5, 3, 2),  # n=6, edges=5, components=3, operations=2
        (6, 4, 3, -1), # n=6, edges=4, components=3, impossible
    ]
    
    print(f"\nExamples:")
    print(f"{'N':<3} {'Edges':<6} {'Components':<11} {'Operations':<11} {'Result'}")
    print(f"{'-'*3} {'-'*6} {'-'*11} {'-'*11} {'-'*6}")
    
    for n, edges, components, operations in examples:
        min_needed = n - 1
        result = "✓" if edges >= min_needed and operations >= 0 else "✗"
        print(f"{n:<3} {edges:<6} {components:<11} {operations:<11} {result}")

if __name__ == "__main__":
    test_make_connected()
    demonstrate_network_analysis()
    analyze_minimum_spanning_tree_connection()

"""
Graph Theory Concepts:
1. Connected Components
2. Union-Find Data Structure
3. Minimum Spanning Tree principles
4. Graph Connectivity Analysis

Key Mathematical Insights:
- Minimum edges for connected graph: n-1
- Operations needed = Connected Components - 1
- Redundant edges can be relocated to bridge components
- Problem reduces to component counting

Algorithm Comparison:
┌─────────────────┬─────────────┬─────────────┬─────────────────────┐
│ Approach        │ Time        │ Space       │ Best Use Case       │
├─────────────────┼─────────────┼─────────────┼─────────────────────┤
│ Union-Find      │ O(E*α(N))   │ O(N)        │ Dynamic connectivity│
│ DFS Components  │ O(N + E)    │ O(N + E)    │ Graph traversal     │
│ BFS Components  │ O(N + E)    │ O(N + E)    │ Level-wise analysis │
└─────────────────┴─────────────┴─────────────┴─────────────────────┘

Real-world Applications:
- Network infrastructure planning
- Social network analysis
- Electrical grid design
- Transportation system optimization
- Communication network resilience
"""
