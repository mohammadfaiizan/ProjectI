"""
2127. Maximum Employees to Be Invited to a Meeting - Multiple Approaches
Difficulty: Medium

A company is organizing a meeting and has a list of n employees, numbered from 0 to n - 1. 
Each employee has a favorite person in the company, and they will only attend the meeting 
if they can sit next to their favorite person.

Given a 0-indexed integer array favorite, where favorite[i] denotes the favorite person 
of the ith employee, return the maximum number of employees that can be invited to the meeting.
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque

class MaximumEmployeesToMeeting:
    """Multiple approaches to solve maximum employees meeting problem"""
    
    def maximumInvitations_functional_graph_analysis(self, favorite: List[int]) -> int:
        """
        Approach 1: Functional Graph Analysis
        
        Analyze the functional graph structure to find optimal solution.
        
        Time: O(n), Space: O(n)
        """
        n = len(favorite)
        
        # Build reverse graph (who likes each person)
        reverse_graph = defaultdict(list)
        for i in range(n):
            reverse_graph[favorite[i]].append(i)
        
        visited = [False] * n
        max_cycle_size = 0
        total_chain_contribution = 0
        
        for start in range(n):
            if visited[start]:
                continue
            
            # Find cycle starting from this node
            cycle_nodes = []
            current = start
            node_to_index = {}
            
            while current not in node_to_index and not visited[current]:
                node_to_index[current] = len(cycle_nodes)
                cycle_nodes.append(current)
                current = favorite[current]
            
            if not visited[current]:
                # Found a new cycle
                cycle_start_index = node_to_index[current]
                cycle = cycle_nodes[cycle_start_index:]
                cycle_size = len(cycle)
                
                # Mark all nodes in path as visited
                for node in cycle_nodes:
                    visited[node] = True
                
                if cycle_size == 2:
                    # Two-node cycle: can extend with chains
                    a, b = cycle[0], cycle[1]
                    chain_a = self._find_longest_chain(reverse_graph, a, favorite, set(cycle))
                    chain_b = self._find_longest_chain(reverse_graph, b, favorite, set(cycle))
                    total_chain_contribution += 2 + chain_a + chain_b
                else:
                    # Larger cycle: take maximum
                    max_cycle_size = max(max_cycle_size, cycle_size)
            else:
                # Mark path nodes as visited
                for node in cycle_nodes:
                    visited[node] = True
        
        return max(max_cycle_size, total_chain_contribution)
    
    def maximumInvitations_cycle_detection_dfs(self, favorite: List[int]) -> int:
        """
        Approach 2: Cycle Detection with DFS
        
        Use DFS to detect cycles and compute chain lengths.
        
        Time: O(n), Space: O(n)
        """
        n = len(favorite)
        in_degree = [0] * n
        
        # Calculate in-degrees
        for fav in favorite:
            in_degree[fav] += 1
        
        # Build reverse graph
        reverse_graph = defaultdict(list)
        for i in range(n):
            reverse_graph[favorite[i]].append(i)
        
        # Find nodes not in any cycle (in-degree 0 after topological sort)
        queue = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)
        
        # Remove nodes not in cycles
        while queue:
            node = queue.popleft()
            next_node = favorite[node]
            in_degree[next_node] -= 1
            if in_degree[next_node] == 0:
                queue.append(next_node)
        
        # Find cycles and calculate contributions
        visited = [False] * n
        max_cycle = 0
        total_pairs = 0
        
        for i in range(n):
            if in_degree[i] > 0 and not visited[i]:
                # This node is part of a cycle
                cycle_size = 0
                current = i
                
                while not visited[current]:
                    visited[current] = True
                    current = favorite[current]
                    cycle_size += 1
                
                if cycle_size == 2:
                    # Two-node cycle
                    node1, node2 = i, favorite[i]
                    chain1 = self._bfs_chain_length(reverse_graph, node1, {node1, node2})
                    chain2 = self._bfs_chain_length(reverse_graph, node2, {node1, node2})
                    total_pairs += 2 + chain1 + chain2
                else:
                    max_cycle = max(max_cycle, cycle_size)
        
        return max(max_cycle, total_pairs)
    
    def maximumInvitations_tarjan_scc(self, favorite: List[int]) -> int:
        """
        Approach 3: Tarjan's Algorithm for SCC
        
        Use Tarjan's algorithm to find strongly connected components.
        
        Time: O(n), Space: O(n)
        """
        n = len(favorite)
        
        # Tarjan's algorithm variables
        index_counter = [0]
        stack = []
        lowlinks = [0] * n
        index = [0] * n
        on_stack = [False] * n
        index_initialized = [False] * n
        sccs = []
        
        def strongconnect(v):
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            index_initialized[v] = True
            stack.append(v)
            on_stack[v] = True
            
            # Consider successor
            w = favorite[v]
            if not index_initialized[w]:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], index[w])
            
            # If v is a root node, pop the stack and create SCC
            if lowlinks[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)
        
        for v in range(n):
            if not index_initialized[v]:
                strongconnect(v)
        
        # Analyze SCCs
        reverse_graph = defaultdict(list)
        for i in range(n):
            reverse_graph[favorite[i]].append(i)
        
        max_cycle = 0
        total_pairs = 0
        
        for scc in sccs:
            if len(scc) == 1:
                # Self-loop or single node
                node = scc[0]
                if favorite[node] == node:
                    max_cycle = max(max_cycle, 1)
            elif len(scc) == 2:
                # Two-node cycle
                node1, node2 = scc[0], scc[1]
                chain1 = self._bfs_chain_length(reverse_graph, node1, set(scc))
                chain2 = self._bfs_chain_length(reverse_graph, node2, set(scc))
                total_pairs += 2 + chain1 + chain2
            else:
                # Larger cycle
                max_cycle = max(max_cycle, len(scc))
        
        return max(max_cycle, total_pairs)
    
    def maximumInvitations_simulation_approach(self, favorite: List[int]) -> int:
        """
        Approach 4: Simulation-Based Approach
        
        Simulate the graph traversal to find cycles and chains.
        
        Time: O(n), Space: O(n)
        """
        n = len(favorite)
        visited = [False] * n
        
        # Build incoming edges
        incoming = defaultdict(list)
        for i in range(n):
            incoming[favorite[i]].append(i)
        
        max_single_cycle = 0
        total_two_cycles = 0
        
        for start in range(n):
            if visited[start]:
                continue
            
            # Trace path to find cycle
            path = []
            current = start
            positions = {}
            
            while current not in positions and not visited[current]:
                positions[current] = len(path)
                path.append(current)
                current = favorite[current]
            
            if not visited[current]:
                # Found new cycle
                cycle_start = positions[current]
                cycle = path[cycle_start:]
                
                # Mark all path nodes as visited
                for node in path:
                    visited[node] = True
                
                if len(cycle) == 2:
                    # Two-node cycle: add chain contributions
                    a, b = cycle
                    chain_a = self._dfs_chain_length(incoming, a, set(cycle))
                    chain_b = self._dfs_chain_length(incoming, b, set(cycle))
                    total_two_cycles += 2 + chain_a + chain_b
                else:
                    # Single large cycle
                    max_single_cycle = max(max_single_cycle, len(cycle))
            else:
                # Mark path nodes as visited
                for node in path:
                    visited[node] = True
        
        return max(max_single_cycle, total_two_cycles)
    
    def _find_longest_chain(self, reverse_graph: Dict, start: int, 
                           favorite: List[int], cycle_nodes: Set[int]) -> int:
        """Find longest chain extending from start node"""
        max_length = 0
        
        def dfs(node, length):
            nonlocal max_length
            max_length = max(max_length, length)
            
            for neighbor in reverse_graph[node]:
                if neighbor not in cycle_nodes:
                    dfs(neighbor, length + 1)
        
        for neighbor in reverse_graph[start]:
            if neighbor not in cycle_nodes:
                dfs(neighbor, 1)
        
        return max_length
    
    def _bfs_chain_length(self, reverse_graph: Dict, start: int, cycle_nodes: Set[int]) -> int:
        """Find longest chain using BFS"""
        max_length = 0
        queue = deque([(neighbor, 1) for neighbor in reverse_graph[start] 
                      if neighbor not in cycle_nodes])
        
        while queue:
            node, length = queue.popleft()
            max_length = max(max_length, length)
            
            for neighbor in reverse_graph[node]:
                if neighbor not in cycle_nodes:
                    queue.append((neighbor, length + 1))
        
        return max_length
    
    def _dfs_chain_length(self, incoming: Dict, start: int, cycle_nodes: Set[int]) -> int:
        """Find longest chain using DFS"""
        max_length = 0
        
        def dfs(node, length):
            nonlocal max_length
            max_length = max(max_length, length)
            
            for neighbor in incoming[node]:
                if neighbor not in cycle_nodes:
                    dfs(neighbor, length + 1)
        
        for neighbor in incoming[start]:
            if neighbor not in cycle_nodes:
                dfs(neighbor, 1)
        
        return max_length

def test_maximum_employees():
    """Test maximum employees algorithms"""
    solver = MaximumEmployeesToMeeting()
    
    test_cases = [
        ([2,2,1,2], 3, "Cycle of 3"),
        ([1,2,0], 3, "Cycle of 3"),
        ([3,0,1,4,1], 4, "Mixed cycles"),
        ([1,0,3,2], 4, "Two 2-cycles"),
        ([1,0,0,2,1,4,7,8,9,6,7,10,8], 6, "Complex case"),
    ]
    
    algorithms = [
        ("Functional Graph", solver.maximumInvitations_functional_graph_analysis),
        ("Cycle Detection DFS", solver.maximumInvitations_cycle_detection_dfs),
        ("Tarjan SCC", solver.maximumInvitations_tarjan_scc),
        ("Simulation", solver.maximumInvitations_simulation_approach),
    ]
    
    print("=== Testing Maximum Employees to Meeting ===")
    
    for favorite, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Favorite: {favorite}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(favorite)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_maximum_employees()

"""
Maximum Employees Meeting problem demonstrates advanced functional graph
analysis with cycle detection and chain optimization techniques.
"""
