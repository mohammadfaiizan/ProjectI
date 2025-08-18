"""
133. Clone Graph
Difficulty: Medium

Problem:
Given a reference of a node in a connected undirected graph.
Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}

Test case format:
For simplicity, each node's value is the same as the node's index (1-indexed). 
For example, the first node with val == 1, the second node with val == 2, and so on. 
The graph is represented in the test case using an adjacency list.

Examples:
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]

Input: adjList = [[]]
Output: [[]]

Input: adjList = []
Output: []

Constraints:
- The number of nodes in the graph is in the range [0, 100]
- 1 <= Node.val <= 100
- Node.val is unique for each node
- There are no repeated edges and no self-loops in the graph
- The Graph is connected and all nodes can be visited starting from the given node
"""

from typing import Dict, Optional, List
from collections import deque

# Definition for a Node
class Node:
    def __init__(self, val: int = 0, neighbors: Optional[List['Node']] = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
    
    def __repr__(self):
        return f"Node({self.val}, neighbors=[{[n.val for n in self.neighbors]}])"

class Solution:
    def cloneGraph_approach1_dfs_recursive(self, node: Optional['Node']) -> Optional['Node']:
        """
        Approach 1: DFS with Recursion and HashMap
        
        Use DFS to traverse the graph and create clones.
        Use a hashmap to track already cloned nodes to handle cycles.
        
        Time: O(V + E) - visit each node and edge once
        Space: O(V) - for the hashmap and recursion stack
        """
        if not node:
            return None
        
        # Map from original node to cloned node
        cloned = {}
        
        def dfs(original_node):
            if original_node in cloned:
                return cloned[original_node]
            
            # Create clone of current node
            clone = Node(original_node.val)
            cloned[original_node] = clone
            
            # Recursively clone all neighbors
            for neighbor in original_node.neighbors:
                clone.neighbors.append(dfs(neighbor))
            
            return clone
        
        return dfs(node)
    
    def cloneGraph_approach2_dfs_iterative(self, node: Optional['Node']) -> Optional['Node']:
        """
        Approach 2: DFS with Iteration and Stack
        
        Use iterative DFS with explicit stack to avoid recursion.
        
        Time: O(V + E)
        Space: O(V)
        """
        if not node:
            return None
        
        cloned = {}
        stack = [node]
        
        # First pass: create all nodes
        while stack:
            current = stack.pop()
            
            if current not in cloned:
                cloned[current] = Node(current.val)
                
                # Add neighbors to stack
                for neighbor in current.neighbors:
                    if neighbor not in cloned:
                        stack.append(neighbor)
        
        # Second pass: connect neighbors
        visited = set()
        stack = [node]
        
        while stack:
            current = stack.pop()
            
            if current not in visited:
                visited.add(current)
                clone = cloned[current]
                
                # Connect cloned neighbors
                for neighbor in current.neighbors:
                    clone.neighbors.append(cloned[neighbor])
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return cloned[node]
    
    def cloneGraph_approach3_bfs(self, node: Optional['Node']) -> Optional['Node']:
        """
        Approach 3: BFS with Queue
        
        Use BFS to traverse and clone the graph level by level.
        
        Time: O(V + E)
        Space: O(V)
        """
        if not node:
            return None
        
        cloned = {}
        queue = deque([node])
        
        # Create the first node
        cloned[node] = Node(node.val)
        
        while queue:
            current = queue.popleft()
            
            # Process all neighbors
            for neighbor in current.neighbors:
                if neighbor not in cloned:
                    # Create clone for new neighbor
                    cloned[neighbor] = Node(neighbor.val)
                    queue.append(neighbor)
                
                # Add neighbor to current clone's neighbor list
                cloned[current].neighbors.append(cloned[neighbor])
        
        return cloned[node]
    
    def cloneGraph_approach4_single_pass_dfs(self, node: Optional['Node']) -> Optional['Node']:
        """
        Approach 4: Single-pass DFS (Most Elegant)
        
        Clone nodes and connect neighbors in a single DFS pass.
        
        Time: O(V + E)
        Space: O(V)
        """
        def dfs(original_node, cloned_map):
            if not original_node:
                return None
            
            if original_node in cloned_map:
                return cloned_map[original_node]
            
            # Create clone
            clone = Node(original_node.val)
            cloned_map[original_node] = clone
            
            # Clone and connect neighbors
            clone.neighbors = [dfs(neighbor, cloned_map) for neighbor in original_node.neighbors]
            
            return clone
        
        return dfs(node, {}) if node else None
    
    def cloneGraph_approach5_two_phase(self, node: Optional['Node']) -> Optional['Node']:
        """
        Approach 5: Two-phase approach (Clear separation)
        
        Phase 1: Create all nodes
        Phase 2: Connect all edges
        
        Time: O(V + E)
        Space: O(V)
        """
        if not node:
            return None
        
        # Phase 1: Discover all nodes and create clones
        cloned = {}
        visited = set()
        stack = [node]
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                cloned[current] = Node(current.val)
                
                for neighbor in current.neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        # Phase 2: Connect edges
        for original_node, cloned_node in cloned.items():
            for neighbor in original_node.neighbors:
                cloned_node.neighbors.append(cloned[neighbor])
        
        return cloned[node]

def build_graph_from_adjacency_list(adj_list: List[List[int]]) -> Optional[Node]:
    """
    Helper function to build a graph from adjacency list representation
    """
    if not adj_list:
        return None
    
    # Create all nodes
    nodes = {}
    for i in range(len(adj_list)):
        nodes[i + 1] = Node(i + 1)
    
    # Connect neighbors
    for i, neighbors in enumerate(adj_list):
        node_val = i + 1
        for neighbor_val in neighbors:
            nodes[node_val].neighbors.append(nodes[neighbor_val])
    
    return nodes[1] if nodes else None

def graph_to_adjacency_list(node: Optional[Node]) -> List[List[int]]:
    """
    Helper function to convert graph back to adjacency list for verification
    """
    if not node:
        return []
    
    visited = set()
    adj_list = {}
    
    def dfs(current_node):
        if current_node.val in visited:
            return
        
        visited.add(current_node.val)
        adj_list[current_node.val] = [neighbor.val for neighbor in current_node.neighbors]
        
        for neighbor in current_node.neighbors:
            dfs(neighbor)
    
    dfs(node)
    
    # Convert to list format
    if not adj_list:
        return []
    
    max_val = max(adj_list.keys())
    result = []
    for i in range(1, max_val + 1):
        result.append(sorted(adj_list.get(i, [])))
    
    return result

def verify_graph_clone(original: Optional[Node], cloned: Optional[Node]) -> bool:
    """
    Verify that the cloned graph is a deep copy of the original
    """
    if not original and not cloned:
        return True
    if not original or not cloned:
        return False
    
    visited_original = set()
    visited_cloned = set()
    
    def dfs_verify(orig, clone):
        if orig.val != clone.val:
            return False
        
        if orig is clone:  # Same reference means not a deep copy
            return False
        
        if orig.val in visited_original:
            return clone.val in visited_cloned
        
        visited_original.add(orig.val)
        visited_cloned.add(clone.val)
        
        if len(orig.neighbors) != len(clone.neighbors):
            return False
        
        # Create mapping of neighbor values
        orig_neighbors = sorted([n.val for n in orig.neighbors])
        clone_neighbors = sorted([n.val for n in clone.neighbors])
        
        if orig_neighbors != clone_neighbors:
            return False
        
        # Recursively verify neighbors
        orig_neighbor_map = {n.val: n for n in orig.neighbors}
        clone_neighbor_map = {n.val: n for n in clone.neighbors}
        
        for val in orig_neighbors:
            if not dfs_verify(orig_neighbor_map[val], clone_neighbor_map[val]):
                return False
        
        return True
    
    return dfs_verify(original, cloned)

def test_clone_graph():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (adjacency_list, description)
        ([[2,4],[1,3],[2,4],[1,3]], "Square graph"),
        ([[]], "Single node"),
        ([], "Empty graph"),
        ([[2],[1]], "Two connected nodes"),
        ([[2,3],[1,3],[1,2]], "Triangle"),
        ([[2,3,4],[1,4],[1,4],[1,2,3]], "Complete graph K4"),
    ]
    
    approaches = [
        ("DFS Recursive", solution.cloneGraph_approach1_dfs_recursive),
        ("DFS Iterative", solution.cloneGraph_approach2_dfs_iterative),
        ("BFS", solution.cloneGraph_approach3_bfs),
        ("Single-pass DFS", solution.cloneGraph_approach4_single_pass_dfs),
        ("Two-phase", solution.cloneGraph_approach5_two_phase),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        
        for i, (adj_list, description) in enumerate(test_cases):
            # Build original graph
            original = build_graph_from_adjacency_list(adj_list)
            
            # Clone the graph
            cloned = func(original)
            
            # Verify the clone
            is_valid_clone = verify_graph_clone(original, cloned)
            
            # Convert back to adjacency list for comparison
            cloned_adj_list = graph_to_adjacency_list(cloned)
            
            status = "✓" if is_valid_clone and cloned_adj_list == adj_list else "✗"
            print(f"Test {i+1}: {status} {description}")
            print(f"         Input: {adj_list}")
            print(f"         Output: {cloned_adj_list}")
            print(f"         Valid clone: {is_valid_clone}")

if __name__ == "__main__":
    test_clone_graph()

"""
Graph Theory Concepts:
1. Graph Traversal (DFS/BFS)
2. Deep Copy vs Shallow Copy
3. Cycle Detection and Handling
4. Graph Representation and Reconstruction

Key Insights:
- Must handle cycles properly using visited tracking
- Need to maintain object identity mapping (original -> clone)
- Multiple traversal strategies possible (DFS recursive/iterative, BFS)
- Verification requires checking both structure and object independence

Algorithm Comparison:
┌─────────────────┬─────────────┬─────────────┬──────────────────┐
│ Approach        │ Time        │ Space       │ Characteristics  │
├─────────────────┼─────────────┼─────────────┼──────────────────┤
│ DFS Recursive   │ O(V + E)    │ O(V + H)    │ Clean, natural   │
│ DFS Iterative   │ O(V + E)    │ O(V)        │ No recursion     │
│ BFS             │ O(V + E)    │ O(V)        │ Level-by-level   │
│ Single-pass DFS │ O(V + E)    │ O(V + H)    │ Most elegant     │
│ Two-phase       │ O(V + E)    │ O(V)        │ Clear separation │
└─────────────────┴─────────────┴─────────────┴──────────────────┘

Real-world Applications:
- Object serialization/deserialization
- Graph database operations
- Network topology replication
- Social network analysis
- State machine duplication
"""
