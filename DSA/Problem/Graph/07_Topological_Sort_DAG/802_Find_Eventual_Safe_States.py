"""
802. Find Eventual Safe States - Multiple Approaches
Difficulty: Medium

There is a directed graph of n nodes with each node labeled from 0 to n - 1. The graph is represented by a 0-indexed 2D integer array graph where graph[i] is an integer array of nodes adjacent to node i, meaning there is an edge from node i to each node in graph[i].

A node is a terminal node if there are no outgoing edges. A node is a safe node if every possible path starting from that node leads to a terminal node (or another safe node).

Return an array containing all the safe nodes of the graph. The answer should be sorted in ascending order.
"""

from typing import List, Set
from collections import defaultdict, deque

class FindEventualSafeStates:
    """Multiple approaches to find eventual safe states"""
    
    def eventualSafeNodes_dfs_cycle_detection(self, graph: List[List[int]]) -> List[int]:
        """
        Approach 1: DFS with Cycle Detection
        
        A node is safe if it doesn't lead to a cycle.
        
        Time: O(V + E), Space: O(V)
        """
        n = len(graph)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        
        def dfs(node: int) -> bool:
            """Returns True if node is safe (doesn't lead to cycle)"""
            if color[node] != WHITE:
                return color[node] == BLACK
            
            color[node] = GRAY
            
            for neighbor in graph[node]:
                if color[neighbor] == GRAY or not dfs(neighbor):
                    return False
            
            color[node] = BLACK
            return True
        
        result = []
        for i in range(n):
            if dfs(i):
                result.append(i)
        
        return result
    
    def eventualSafeNodes_reverse_topological(self, graph: List[List[int]]) -> List[int]:
        """
        Approach 2: Reverse Graph + Topological Sort
        
        Reverse the graph and find nodes reachable from terminal nodes.
        
        Time: O(V + E), Space: O(V + E)
        """
        n = len(graph)
        
        # Build reverse graph
        reverse_graph = defaultdict(list)
        out_degree = [0] * n
        
        for i in range(n):
            out_degree[i] = len(graph[i])
            for neighbor in graph[i]:
                reverse_graph[neighbor].append(i)
        
        # Start from terminal nodes (out_degree = 0)
        queue = deque()
        for i in range(n):
            if out_degree[i] == 0:
                queue.append(i)
        
        safe_nodes = set()
        
        while queue:
            node = queue.popleft()
            safe_nodes.add(node)
            
            # Process nodes that point to current safe node
            for prev_node in reverse_graph[node]:
                out_degree[prev_node] -= 1
                if out_degree[prev_node] == 0:
                    queue.append(prev_node)
        
        return sorted(list(safe_nodes))
    
    def eventualSafeNodes_iterative_dfs(self, graph: List[List[int]]) -> List[int]:
        """
        Approach 3: Iterative DFS with Stack
        
        Use iterative DFS to avoid recursion depth issues.
        
        Time: O(V + E), Space: O(V)
        """
        n = len(graph)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        safe = [False] * n
        
        def is_safe(start: int) -> bool:
            if color[start] != WHITE:
                return color[start] == BLACK
            
            stack = [start]
            path = []
            
            while stack:
                node = stack[-1]
                
                if color[node] == WHITE:
                    color[node] = GRAY
                    path.append(node)
                    
                    # Add unvisited neighbors
                    for neighbor in graph[node]:
                        if color[neighbor] == GRAY:  # Cycle detected
                            return False
                        if color[neighbor] == WHITE:
                            stack.append(neighbor)
                
                elif color[node] == GRAY:
                    # Finished processing this node
                    color[node] = BLACK
                    safe[node] = True
                    stack.pop()
                    if path and path[-1] == node:
                        path.pop()
                
                else:  # BLACK
                    stack.pop()
            
            return True
        
        result = []
        for i in range(n):
            if is_safe(i):
                result.append(i)
        
        return result
    
    def eventualSafeNodes_memoized_dfs(self, graph: List[List[int]]) -> List[int]:
        """
        Approach 4: DFS with Memoization
        
        Use memoization to avoid recomputing safety of nodes.
        
        Time: O(V + E), Space: O(V)
        """
        n = len(graph)
        memo = {}  # -1: unsafe, 1: safe, 0: unknown
        
        def is_safe(node: int, visiting: Set[int]) -> bool:
            if node in memo:
                return memo[node] == 1
            
            if node in visiting:  # Cycle detected
                memo[node] = -1
                return False
            
            visiting.add(node)
            
            # Check all neighbors
            for neighbor in graph[node]:
                if not is_safe(neighbor, visiting):
                    memo[node] = -1
                    visiting.remove(node)
                    return False
            
            visiting.remove(node)
            memo[node] = 1
            return True
        
        result = []
        for i in range(n):
            if is_safe(i, set()):
                result.append(i)
        
        return result
    
    def eventualSafeNodes_tarjan_scc(self, graph: List[List[int]]) -> List[int]:
        """
        Approach 5: Tarjan's SCC Algorithm
        
        Use Tarjan's algorithm to find strongly connected components.
        
        Time: O(V + E), Space: O(V)
        """
        n = len(graph)
        
        # Tarjan's algorithm variables
        ids = [-1] * n
        low = [-1] * n
        on_stack = [False] * n
        stack = []
        id_counter = 0
        scc_count = 0
        
        def tarjan(at: int):
            nonlocal id_counter, scc_count
            
            stack.append(at)
            on_stack[at] = True
            ids[at] = low[at] = id_counter
            id_counter += 1
            
            # Visit neighbors
            for to in graph[at]:
                if ids[to] == -1:
                    tarjan(to)
                if on_stack[to]:
                    low[at] = min(low[at], low[to])
            
            # Found SCC root
            if ids[at] == low[at]:
                while True:
                    node = stack.pop()
                    on_stack[node] = False
                    low[node] = scc_count
                    if node == at:
                        break
                scc_count += 1
        
        # Run Tarjan's algorithm
        for i in range(n):
            if ids[i] == -1:
                tarjan(i)
        
        # A node is safe if it's in an SCC with no outgoing edges to other SCCs
        # or if it only leads to safe SCCs
        scc_has_cycle = [False] * scc_count
        scc_out_edges = [set() for _ in range(scc_count)]
        
        # Check for cycles within SCCs and outgoing edges
        for i in range(n):
            scc_i = low[i]
            for neighbor in graph[i]:
                scc_neighbor = low[neighbor]
                if scc_i == scc_neighbor and i != neighbor:
                    scc_has_cycle[scc_i] = True
                elif scc_i != scc_neighbor:
                    scc_out_edges[scc_i].add(scc_neighbor)
        
        # Find safe SCCs using topological sort
        safe_scc = [False] * scc_count
        
        # Terminal SCCs (no outgoing edges and no cycles) are safe
        for i in range(scc_count):
            if not scc_out_edges[i] and not scc_has_cycle[i]:
                safe_scc[i] = True
        
        # Propagate safety backwards
        changed = True
        while changed:
            changed = False
            for i in range(scc_count):
                if not safe_scc[i] and not scc_has_cycle[i]:
                    if all(safe_scc[j] for j in scc_out_edges[i]):
                        safe_scc[i] = True
                        changed = True
        
        # Collect safe nodes
        result = []
        for i in range(n):
            if safe_scc[low[i]]:
                result.append(i)
        
        return sorted(result)

def test_eventual_safe_states():
    """Test eventual safe states algorithms"""
    solver = FindEventualSafeStates()
    
    test_cases = [
        ([[1,2],[2,3],[5],[0],[5],[],[]], [2,4,5,6], "Example 1"),
        ([[1,2,3,4],[1,2],[3,4],[0,4],[]], [4], "Example 2"),
        ([[],[0,2,3,4],[3],[4],[]], [0,1,2,3,4], "All nodes safe"),
        ([[1],[2],[0]], [], "All nodes in cycle"),
        ([[], [0], [1]], [0,1,2], "Simple terminal nodes"),
    ]
    
    algorithms = [
        ("DFS Cycle Detection", solver.eventualSafeNodes_dfs_cycle_detection),
        ("Reverse Topological", solver.eventualSafeNodes_reverse_topological),
        ("Iterative DFS", solver.eventualSafeNodes_iterative_dfs),
        ("Memoized DFS", solver.eventualSafeNodes_memoized_dfs),
    ]
    
    print("=== Testing Find Eventual Safe States ===")
    
    for graph, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Graph: {graph}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Safe nodes: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_eventual_safe_states()

"""
Find Eventual Safe States demonstrates cycle detection
and safety analysis in directed graphs using DFS,
topological sorting, and strongly connected components.
"""
