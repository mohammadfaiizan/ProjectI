"""
1203. Sort Items by Groups Respecting Dependencies - Multiple Approaches
Difficulty: Hard

There are n items each belonging to zero or one of m groups where group[i] is the group that the i-th item belongs to and it's equal to -1 if the i-th item belongs to no group. The items and the groups are zero indexed. A group can have no item belonging to it.

Return a sorted list of the items such that:
- The items that belong to the same group are next to each other in the sorted list.
- There are some relations between these items where beforeItems[i] is a list containing all the items that should come before the i-th item in the sorted list (to the left of the i-th item).

Return any solution if there is more than one solution and return an empty array if there is no solution.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque

class SortItemsByGroups:
    """Multiple approaches to sort items by groups with dependencies"""
    
    def sortItems_dual_topological_sort(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        Approach 1: Dual Topological Sort
        
        Perform topological sort on both groups and items within groups.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Assign unique group IDs to items with no group (-1)
        group_id = m
        for i in range(n):
            if group[i] == -1:
                group[i] = group_id
                group_id += 1
        
        # Build group dependency graph
        group_graph = defaultdict(set)
        group_in_degree = defaultdict(int)
        
        # Build item dependency graph
        item_graph = defaultdict(list)
        item_in_degree = [0] * n
        
        # Initialize in-degrees
        for i in range(group_id):
            group_in_degree[i] = 0
        
        # Process dependencies
        for i in range(n):
            for before_item in beforeItems[i]:
                # Item dependency
                item_graph[before_item].append(i)
                item_in_degree[i] += 1
                
                # Group dependency
                before_group = group[before_item]
                current_group = group[i]
                
                if before_group != current_group:
                    if current_group not in group_graph[before_group]:
                        group_graph[before_group].add(current_group)
                        group_in_degree[current_group] += 1
        
        # Topological sort for groups
        def topological_sort_groups():
            queue = deque()
            for g in range(group_id):
                if group_in_degree[g] == 0:
                    queue.append(g)
            
            group_order = []
            while queue:
                current_group = queue.popleft()
                group_order.append(current_group)
                
                for next_group in group_graph[current_group]:
                    group_in_degree[next_group] -= 1
                    if group_in_degree[next_group] == 0:
                        queue.append(next_group)
            
            return group_order if len(group_order) == group_id else []
        
        # Topological sort for items within each group
        def topological_sort_items():
            queue = deque()
            for i in range(n):
                if item_in_degree[i] == 0:
                    queue.append(i)
            
            item_order = []
            while queue:
                item = queue.popleft()
                item_order.append(item)
                
                for next_item in item_graph[item]:
                    item_in_degree[next_item] -= 1
                    if item_in_degree[next_item] == 0:
                        queue.append(next_item)
            
            return item_order if len(item_order) == n else []
        
        # Get topological orders
        group_order = topological_sort_groups()
        item_order = topological_sort_items()
        
        if not group_order or not item_order:
            return []
        
        # Group items by their groups
        group_to_items = defaultdict(list)
        for item in item_order:
            group_to_items[group[item]].append(item)
        
        # Build final result
        result = []
        for g in group_order:
            result.extend(group_to_items[g])
        
        return result
    
    def sortItems_unified_topological_sort(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        Approach 2: Unified Topological Sort
        
        Create a unified graph with both items and groups as nodes.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Assign groups to ungrouped items
        next_group_id = m
        for i in range(n):
            if group[i] == -1:
                group[i] = next_group_id
                next_group_id += 1
        
        total_groups = next_group_id
        
        # Create unified graph: items [0, n-1], groups [n, n+total_groups-1]
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Initialize in-degrees
        for i in range(n + total_groups):
            in_degree[i] = 0
        
        # Add edges: group -> item (each item depends on its group)
        for i in range(n):
            group_node = n + group[i]
            graph[group_node].append(i)
            in_degree[i] += 1
        
        # Add item dependencies
        for i in range(n):
            for before_item in beforeItems[i]:
                if group[before_item] == group[i]:
                    # Same group: direct item dependency
                    graph[before_item].append(i)
                    in_degree[i] += 1
                else:
                    # Different groups: group dependency
                    before_group_node = n + group[before_item]
                    current_group_node = n + group[i]
                    graph[before_group_node].append(current_group_node)
                    in_degree[current_group_node] += 1
        
        # Topological sort
        queue = deque()
        for i in range(n + total_groups):
            if in_degree[i] == 0:
                queue.append(i)
        
        topo_order = []
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if topological sort is complete
        if len(topo_order) != n + total_groups:
            return []
        
        # Extract items in order
        result = []
        for node in topo_order:
            if node < n:  # It's an item
                result.append(node)
        
        return result
    
    def sortItems_hierarchical_approach(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        Approach 3: Hierarchical Topological Sort
        
        Sort groups first, then items within each group.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Handle ungrouped items
        group_counter = m
        for i in range(n):
            if group[i] == -1:
                group[i] = group_counter
                group_counter += 1
        
        # Build graphs
        item_graph = [[] for _ in range(n)]
        item_indegree = [0] * n
        
        group_graph = [[] for _ in range(group_counter)]
        group_indegree = [0] * group_counter
        
        # Process dependencies
        for curr in range(n):
            for prev in beforeItems[curr]:
                item_graph[prev].append(curr)
                item_indegree[curr] += 1
                
                if group[prev] != group[curr]:
                    group_graph[group[prev]].append(group[curr])
                    group_indegree[group[curr]] += 1
        
        # Remove duplicate group edges
        for i in range(group_counter):
            group_graph[i] = list(set(group_graph[i]))
        
        # Recalculate group in-degrees
        group_indegree = [0] * group_counter
        for i in range(group_counter):
            for j in group_graph[i]:
                group_indegree[j] += 1
        
        def topological_sort(graph, indegree):
            """Generic topological sort"""
            queue = deque()
            for i in range(len(indegree)):
                if indegree[i] == 0:
                    queue.append(i)
            
            result = []
            while queue:
                node = queue.popleft()
                result.append(node)
                
                for neighbor in graph[node]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)
            
            return result if len(result) == len(indegree) else []
        
        # Sort groups
        group_order = topological_sort(group_graph, group_indegree[:])
        if not group_order:
            return []
        
        # Sort items
        item_order = topological_sort(item_graph, item_indegree[:])
        if not item_order:
            return []
        
        # Group items by their groups
        grouped_items = [[] for _ in range(group_counter)]
        for item in item_order:
            grouped_items[group[item]].append(item)
        
        # Build result following group order
        result = []
        for g in group_order:
            result.extend(grouped_items[g])
        
        return result
    
    def sortItems_dfs_approach(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        Approach 4: DFS-based Topological Sort
        
        Use DFS for topological sorting with cycle detection.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Assign groups to ungrouped items
        group_id = m
        for i in range(n):
            if group[i] == -1:
                group[i] = group_id
                group_id += 1
        
        # Build dependency graphs
        item_deps = [[] for _ in range(n)]
        group_deps = [[] for _ in range(group_id)]
        
        for i in range(n):
            for j in beforeItems[i]:
                item_deps[j].append(i)
                if group[j] != group[i]:
                    group_deps[group[j]].append(group[i])
        
        # Remove duplicate group dependencies
        for i in range(group_id):
            group_deps[i] = list(set(group_deps[i]))
        
        # DFS topological sort
        def dfs_topo_sort(graph):
            WHITE, GRAY, BLACK = 0, 1, 2
            color = [WHITE] * len(graph)
            result = []
            
            def dfs(node):
                if color[node] == GRAY:  # Cycle detected
                    return False
                if color[node] == BLACK:  # Already processed
                    return True
                
                color[node] = GRAY
                for neighbor in graph[node]:
                    if not dfs(neighbor):
                        return False
                
                color[node] = BLACK
                result.append(node)
                return True
            
            for i in range(len(graph)):
                if color[i] == WHITE:
                    if not dfs(i):
                        return []
            
            return result[::-1]  # Reverse for correct order
        
        # Sort groups and items
        group_order = dfs_topo_sort(group_deps)
        item_order = dfs_topo_sort(item_deps)
        
        if not group_order or not item_order:
            return []
        
        # Organize items by groups
        group_items = [[] for _ in range(group_id)]
        for item in item_order:
            group_items[group[item]].append(item)
        
        # Build final result
        result = []
        for g in group_order:
            result.extend(group_items[g])
        
        return result
    
    def sortItems_optimized_kahn(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        Approach 5: Optimized Kahn's Algorithm
        
        Optimized implementation of Kahn's algorithm for both levels.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Handle ungrouped items
        next_group = m
        for i in range(n):
            if group[i] == -1:
                group[i] = next_group
                next_group += 1
        
        # Build adjacency lists and in-degrees
        item_adj = [[] for _ in range(n)]
        item_in = [0] * n
        
        group_adj = [set() for _ in range(next_group)]
        group_in = [0] * next_group
        
        # Process all dependencies
        for item in range(n):
            for prereq in beforeItems[item]:
                # Add item dependency
                item_adj[prereq].append(item)
                item_in[item] += 1
                
                # Add group dependency if different groups
                if group[prereq] != group[item]:
                    if group[item] not in group_adj[group[prereq]]:
                        group_adj[group[prereq]].add(group[item])
                        group_in[group[item]] += 1
        
        # Convert sets to lists for group adjacency
        for i in range(next_group):
            group_adj[i] = list(group_adj[i])
        
        # Kahn's algorithm for topological sort
        def kahn_sort(adj, in_deg):
            queue = deque()
            for i in range(len(in_deg)):
                if in_deg[i] == 0:
                    queue.append(i)
            
            order = []
            while queue:
                node = queue.popleft()
                order.append(node)
                
                for neighbor in adj[node]:
                    in_deg[neighbor] -= 1
                    if in_deg[neighbor] == 0:
                        queue.append(neighbor)
            
            return order if len(order) == len(in_deg) else []
        
        # Get topological orders
        group_order = kahn_sort(group_adj, group_in[:])
        item_order = kahn_sort(item_adj, item_in[:])
        
        if not group_order or not item_order:
            return []
        
        # Group items by their group IDs
        items_by_group = [[] for _ in range(next_group)]
        for item in item_order:
            items_by_group[group[item]].append(item)
        
        # Construct final result
        result = []
        for grp in group_order:
            result.extend(items_by_group[grp])
        
        return result

def test_sort_items_by_groups():
    """Test sort items by groups algorithms"""
    solver = SortItemsByGroups()
    
    test_cases = [
        (8, 2, [-1,-1,1,0,0,1,0,-1], [[],[6],[5],[6],[3,6],[],[],[]], "Example 1"),
        (8, 2, [-1,-1,1,0,0,1,0,-1], [[],[6],[5],[6],[3],[],[4],[]], "Example 2 - impossible"),
        (5, 3, [0,0,1,1,2], [[],[0],[],[2],[1]], "Simple case"),
        (3, 1, [0,0,-1], [[],[],[1,2]], "Mixed groups"),
    ]
    
    algorithms = [
        ("Dual Topological Sort", solver.sortItems_dual_topological_sort),
        ("Unified Topological Sort", solver.sortItems_unified_topological_sort),
        ("Hierarchical Approach", solver.sortItems_hierarchical_approach),
        ("DFS Approach", solver.sortItems_dfs_approach),
        ("Optimized Kahn", solver.sortItems_optimized_kahn),
    ]
    
    print("=== Testing Sort Items by Groups ===")
    
    for n, m, group, beforeItems, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"n={n}, m={m}, group={group}")
        print(f"beforeItems={beforeItems}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, m, group[:], beforeItems)
                if result:
                    print(f"{alg_name:22} | ✓ | Result: {result}")
                else:
                    print(f"{alg_name:22} | ✓ | No solution (empty array)")
            except Exception as e:
                print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_sort_items_by_groups()

"""
Sort Items by Groups demonstrates advanced topological sorting
with hierarchical constraints and dual-level dependency management
for complex scheduling problems.
"""
