"""
1791. Find Center of Star Graph
Difficulty: Easy

Problem:
There is an undirected star graph consisting of n nodes labeled from 1 to n. 
A star graph is a graph where there is one center node and exactly n - 1 edges 
that connect the center node with every other node.

You are given a 2D integer array edges where each edges[i] = [ui, vi] indicates 
that there is an edge between the nodes ui and vi. Return the center of the star graph.

Examples:
Input: edges = [[1,2],[2,3],[4,2]]
Output: 2

Input: edges = [[1,2],[5,1],[1,3],[1,4]]
Output: 1

Constraints:
- 3 <= n <= 10^5
- edges.length == n - 1
- edges[i].length == 2
- 1 <= ui, vi <= n
- ui != vi
- The given edges represent a valid star graph
"""

from typing import List
from collections import defaultdict, Counter

class Solution:
    def findCenter_approach1_degree_counting(self, edges: List[List[int]]) -> int:
        """
        Approach 1: Degree counting - Complete scan
        
        In a star graph, the center node has degree n-1,
        while all other nodes have degree 1.
        
        Time: O(E) = O(N) where E = number of edges = n-1
        Space: O(N) for degree counting
        """
        degree = defaultdict(int)
        
        # Count degree of each node
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        
        # Find node with maximum degree (should be n-1)
        n = len(edges) + 1  # Number of nodes
        for node, deg in degree.items():
            if deg == n - 1:
                return node
        
        return -1  # Should never reach here for valid input
    
    def findCenter_approach2_optimized_check(self, edges: List[List[int]]) -> int:
        """
        Approach 2: Optimized - Check first two edges only
        
        Key insight: In a star graph, the center node must appear
        in EVERY edge. So it must appear in both the first and 
        second edge.
        
        Time: O(1) - Only checks first two edges
        Space: O(1)
        """
        # The center must be in both first and second edge
        first_edge = edges[0]
        second_edge = edges[1]
        
        # Find common node between first two edges
        if first_edge[0] in second_edge:
            return first_edge[0]
        else:
            return first_edge[1]
    
    def findCenter_approach3_counter_method(self, edges: List[List[int]]) -> int:
        """
        Approach 3: Counter-based approach
        
        Count frequency of each node in edges.
        Center appears in all n-1 edges, others appear in only 1.
        
        Time: O(E) = O(N)
        Space: O(N)
        """
        node_count = Counter()
        
        for u, v in edges:
            node_count[u] += 1
            node_count[v] += 1
        
        # Return node with highest frequency
        return node_count.most_common(1)[0][0]
    
    def findCenter_approach4_set_intersection(self, edges: List[List[int]]) -> int:
        """
        Approach 4: Set intersection of first two edges
        
        Center node is the intersection of any two edges.
        
        Time: O(1)
        Space: O(1)
        """
        return list(set(edges[0]) & set(edges[1]))[0]
    
    def findCenter_approach5_mathematical(self, edges: List[List[int]]) -> int:
        """
        Approach 5: Mathematical approach using sum
        
        If we know the total sum of all nodes and subtract
        the sum excluding duplicates, we can find the center.
        
        Time: O(E) = O(N)
        Space: O(N)
        """
        all_nodes = set()
        total_sum = 0
        
        for u, v in edges:
            all_nodes.add(u)
            all_nodes.add(v)
            total_sum += u + v
        
        unique_sum = sum(all_nodes)
        n = len(all_nodes)
        
        # Center appears (n-1) times, others appear once
        # total_sum = unique_sum + center * (n-2)
        center = (total_sum - unique_sum) // (n - 2)
        return center

def test_find_center():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (edges, expected)
        ([[1, 2], [2, 3], [4, 2]], 2),
        ([[1, 2], [5, 1], [1, 3], [1, 4]], 1),
        ([[1, 2], [1, 3]], 1),  # Minimal star (3 nodes)
        ([[2, 1], [3, 2], [2, 4], [2, 5]], 2),  # 5-node star
        ([[3, 1], [3, 2], [3, 4], [3, 5], [3, 6]], 3),  # 6-node star
    ]
    
    approaches = [
        ("Degree Counting", solution.findCenter_approach1_degree_counting),
        ("Optimized Check", solution.findCenter_approach2_optimized_check),
        ("Counter Method", solution.findCenter_approach3_counter_method),
        ("Set Intersection", solution.findCenter_approach4_set_intersection),
        ("Mathematical", solution.findCenter_approach5_mathematical),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (edges, expected) in enumerate(test_cases):
            result = func(edges)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Input: {edges}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_star_graph_properties():
    """Demonstrate key properties of star graphs"""
    print("\n=== Star Graph Properties Demo ===")
    
    # Example: 5-node star with center = 3
    edges = [[3, 1], [3, 2], [3, 4], [3, 5]]
    
    print(f"Edges: {edges}")
    print(f"Number of nodes: {len(edges) + 1}")
    print(f"Number of edges: {len(edges)}")
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    print("\nAdjacency List:")
    for node in sorted(adj.keys()):
        print(f"Node {node}: {adj[node]} (degree: {len(adj[node])})")
    
    # Properties verification
    degrees = [len(adj[node]) for node in adj]
    max_degree = max(degrees)
    print(f"\nMax degree: {max_degree}")
    print(f"Expected center degree: {len(edges)}")
    print(f"Is valid star graph: {max_degree == len(edges)}")

if __name__ == "__main__":
    test_find_center()
    demonstrate_star_graph_properties()

"""
Graph Theory Concepts:
1. Star Graph Structure and Properties
2. Degree Centrality
3. Graph Invariants
4. Connectivity Patterns

Key Properties of Star Graphs:
- Exactly one center node with degree n-1
- All other nodes have degree 1
- Total edges = n-1 (tree structure)
- Diameter = 2 (max distance between any two nodes)
- Center appears in every edge

Optimization Insights:
- Approach 2 is optimal O(1) - leverages star graph property
- Other approaches show different algorithmic perspectives
- Problem demonstrates importance of understanding graph structure

Real-world Applications:
- Network topology design (hub-and-spoke)
- Communication systems (centralized architecture)
- Social network analysis (celebrity/influencer detection)
- Computer networks (star topology)
"""
