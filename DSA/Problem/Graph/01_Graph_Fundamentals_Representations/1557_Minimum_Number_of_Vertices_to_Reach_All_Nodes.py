"""
1557. Minimum Number of Vertices to Reach All Nodes
Difficulty: Easy

Problem:
Given a directed acyclic graph, with n vertices numbered from 0 to n-1, and an array 
edges where edges[i] = [fromi, toi] represents a directed edge from node fromi to node toi.

Find the minimum number of vertices from which all other vertices in the graph are reachable. 
It's guaranteed that a unique solution exists.

Examples:
Input: n = 6, edges = [[0,1],[0,2],[2,3],[3,4],[3,5]]
Output: [0]
Explanation: From node 0 we can reach all nodes in the graph.

Input: n = 5, edges = [[0,1],[2,1],[3,1],[1,4],[2,4]]
Output: [0,2,3]
Explanation: We need all of [0,2,3] to reach all nodes.

Constraints:
- 2 <= n <= 10^5
- 1 <= edges.length <= min(10^5, n*(n-1)/2)
- edges[i].length == 2
- 0 <= fromi != toi < n
- All pairs (fromi, toi) are distinct
- The input graph is a DAG (Directed Acyclic Graph)
"""

from typing import List
from collections import defaultdict, deque

class Solution:
    def findSmallestSetOfVertices_approach1_indegree(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 1: In-degree analysis (Optimal)
        
        Key insight: We need all nodes with in-degree 0.
        These are nodes that cannot be reached from any other node,
        so they must be in our starting set.
        
        Proof: Any node with in-degree > 0 can be reached from some other node,
        so we don't need to include it in our starting set.
        
        Time: O(V + E)
        Space: O(V)
        """
        # Calculate in-degree for each node
        in_degree = [0] * n
        
        for from_node, to_node in edges:
            in_degree[to_node] += 1
        
        # Return all nodes with in-degree 0
        return [node for node in range(n) if in_degree[node] == 0]
    
    def findSmallestSetOfVertices_approach2_reachability(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 2: Reachability analysis using DFS
        
        Try to find minimum set by checking reachability.
        Start with all nodes, then remove those reachable from others.
        
        Time: O(V * (V + E)) - potentially expensive
        Space: O(V + E)
        """
        # Build adjacency list
        adj = defaultdict(list)
        for from_node, to_node in edges:
            adj[from_node].append(to_node)
        
        def can_reach_all_from(start_nodes):
            """Check if we can reach all nodes from given start nodes"""
            visited = set()
            
            def dfs(node):
                if node in visited:
                    return
                visited.add(node)
                for neighbor in adj[node]:
                    dfs(neighbor)
            
            for start in start_nodes:
                dfs(start)
            
            return len(visited) == n
        
        # Find nodes with in-degree 0 (this is actually the optimal solution)
        in_degree = [0] * n
        for from_node, to_node in edges:
            in_degree[to_node] += 1
        
        candidates = [node for node in range(n) if in_degree[node] == 0]
        
        # Verify this set can reach all nodes
        if can_reach_all_from(candidates):
            return candidates
        
        return []  # Should not reach here for valid DAG
    
    def findSmallestSetOfVertices_approach3_set_difference(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 3: Set operations approach
        
        Start with all nodes, then remove those that appear as 'to' nodes
        in edges (i.e., those that can be reached from others).
        
        Time: O(E)
        Space: O(V)
        """
        # All nodes initially
        all_nodes = set(range(n))
        
        # Remove nodes that can be reached (appear as 'to' in edges)
        reachable_nodes = set()
        for from_node, to_node in edges:
            reachable_nodes.add(to_node)
        
        # Return nodes that are not reachable from others
        return list(all_nodes - reachable_nodes)
    
    def findSmallestSetOfVertices_approach4_topological(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 4: Topological sorting perspective
        
        In topological order, nodes with in-degree 0 come first.
        These are exactly the nodes we need.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build graph and calculate in-degrees
        adj = defaultdict(list)
        in_degree = [0] * n
        
        for from_node, to_node in edges:
            adj[from_node].append(to_node)
            in_degree[to_node] += 1
        
        # Find all source nodes (in-degree 0)
        sources = []
        for node in range(n):
            if in_degree[node] == 0:
                sources.append(node)
        
        return sources

def test_find_smallest_set():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, expected)
        (6, [[0,1],[0,2],[2,3],[3,4],[3,5]], [0]),
        (5, [[0,1],[2,1],[3,1],[1,4],[2,4]], [0,2,3]),
        (3, [[1,2]], [0,1]),  # Disconnected components
        (4, [[0,1],[1,2],[2,3]], [0]),  # Linear chain
        (4, [[0,1],[0,2],[0,3]], [0]),  # Star pattern
        (2, [], [0,1]),  # No edges
    ]
    
    approaches = [
        ("In-degree Analysis", solution.findSmallestSetOfVertices_approach1_indegree),
        ("Reachability Check", solution.findSmallestSetOfVertices_approach2_reachability),
        ("Set Difference", solution.findSmallestSetOfVertices_approach3_set_difference),
        ("Topological Perspective", solution.findSmallestSetOfVertices_approach4_topological),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, expected) in enumerate(test_cases):
            result = func(n, edges)
            # Sort for comparison since order doesn't matter
            result_sorted = sorted(result)
            expected_sorted = sorted(expected)
            status = "✓" if result_sorted == expected_sorted else "✗"
            print(f"Test {i+1}: {status} n={n}, edges={edges}")
            print(f"         Expected: {expected_sorted}, Got: {result_sorted}")

def demonstrate_dag_properties():
    """Demonstrate DAG properties and reachability"""
    print("\n=== DAG Properties Demo ===")
    
    # Example DAG
    n = 6
    edges = [[0,1],[0,2],[2,3],[3,4],[3,5]]
    
    print(f"Graph: n={n}, edges={edges}")
    
    # Build adjacency list and calculate in-degrees
    adj = defaultdict(list)
    in_degree = [0] * n
    
    for from_node, to_node in edges:
        adj[from_node].append(to_node)
        in_degree[to_node] += 1
    
    print(f"\nAdjacency List:")
    for node in range(n):
        print(f"Node {node}: {adj[node]} (out-degree: {len(adj[node])})")
    
    print(f"\nIn-degrees: {in_degree}")
    
    # Find sources
    sources = [node for node in range(n) if in_degree[node] == 0]
    print(f"Source nodes (in-degree 0): {sources}")
    
    # Verify reachability from sources
    def dfs_reachable(start):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in adj[node]:
                    stack.append(neighbor)
        return visited
    
    all_reachable = set()
    for source in sources:
        reachable = dfs_reachable(source)
        print(f"Reachable from {source}: {sorted(reachable)}")
        all_reachable.update(reachable)
    
    print(f"Total reachable from sources: {sorted(all_reachable)}")
    print(f"Can reach all nodes: {len(all_reachable) == n}")

if __name__ == "__main__":
    test_find_smallest_set()
    demonstrate_dag_properties()

"""
Graph Theory Concepts:
1. Directed Acyclic Graph (DAG) properties
2. In-degree and reachability analysis
3. Source nodes in DAG
4. Topological ordering concepts

Key Insights:
- In a DAG, nodes with in-degree 0 are essential starting points
- These source nodes cannot be reached from any other node
- Every other node can be reached from at least one source node
- This is related to the concept of "dominators" in graph theory

Mathematical Proof:
- Let S be the set of nodes with in-degree 0
- Any node not in S has at least one incoming edge
- By transitivity, any node not in S can be reached from some node in S
- Therefore, S is the minimum set needed to reach all nodes

Real-world Applications:
- Package dependency resolution (find root packages)
- Task scheduling (find independent starting tasks)
- Code compilation order (find source files)
- Academic prerequisite analysis
"""
