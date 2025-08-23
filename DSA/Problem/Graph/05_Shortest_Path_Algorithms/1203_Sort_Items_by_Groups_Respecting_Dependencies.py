"""
1203. Sort Items by Groups Respecting Dependencies - Multiple Approaches
Difficulty: Hard

There are n items each belonging to zero or one of m groups. You are given the array group where group[i] is the group that the i-th item belongs to and it's equal to -1 if the i-th item belongs to no group. The items and the groups are zero indexed. A group can have no item belonging to it.

Return a sorted list of the items such that:
- The items that belong to the same group are next to each other in the sorted list.
- There are some relations between these items where beforeItems[i] is a list containing all the items that should come before the i-th item in the sorted list (to the left of the i-th item).

Return any solution if there is more than one solution and return an empty array if there is no solution.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque

class SortItemsByGroups:
    """Multiple approaches to sort items with group and dependency constraints"""
    
    def sortItems_dual_topological_sort(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        Approach 1: Dual Topological Sort
        
        Perform topological sort on both groups and items within groups.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Assign unique group IDs to items without groups
        group_id = m
        for i in range(n):
            if group[i] == -1:
                group[i] = group_id
                group_id += 1
        
        # Build group graph and item graph
        group_graph = defaultdict(set)
        group_indegree = defaultdict(int)
        
        item_graph = defaultdict(list)
        item_indegree = [0] * n
        
        # Group items by their groups
        group_items = defaultdict(list)
        for i in range(n):
            group_items[group[i]].append(i)
        
        # Build graphs based on dependencies
        for i in range(n):
            for before_item in beforeItems[i]:
                # Item dependency
                item_graph[before_item].append(i)
                item_indegree[i] += 1
                
                # Group dependency (if different groups)
                if group[before_item] != group[i]:
                    if group[i] not in group_graph[group[before_item]]:
                        group_graph[group[before_item]].add(group[i])
                        group_indegree[group[i]] += 1
        
        # Initialize indegrees for all groups
        for g in group_items:
            if g not in group_indegree:
                group_indegree[g] = 0
        
        # Topological sort for groups
        def topological_sort_groups():
            queue = deque()
            for g in group_indegree:
                if group_indegree[g] == 0:
                    queue.append(g)
            
            group_order = []
            while queue:
                current_group = queue.popleft()
                group_order.append(current_group)
                
                for next_group in group_graph[current_group]:
                    group_indegree[next_group] -= 1
                    if group_indegree[next_group] == 0:
                        queue.append(next_group)
            
            return group_order if len(group_order) == len(group_indegree) else []
        
        # Topological sort for items within each group
        def topological_sort_items_in_group(items):
            if not items:
                return []
            
            # Build subgraph for items in this group
            local_graph = defaultdict(list)
            local_indegree = {item: 0 for item in items}
            
            for item in items:
                for neighbor in item_graph[item]:
                    if neighbor in local_indegree:  # Neighbor is in same group
                        local_graph[item].append(neighbor)
                        local_indegree[neighbor] += 1
            
            # Topological sort
            queue = deque()
            for item in items:
                if local_indegree[item] == 0:
                    queue.append(item)
            
            item_order = []
            while queue:
                current_item = queue.popleft()
                item_order.append(current_item)
                
                for next_item in local_graph[current_item]:
                    local_indegree[next_item] -= 1
                    if local_indegree[next_item] == 0:
                        queue.append(next_item)
            
            return item_order if len(item_order) == len(items) else []
        
        # Get group order
        group_order = topological_sort_groups()
        if not group_order:
            return []
        
        # Sort items within each group and combine
        result = []
        for g in group_order:
            items_in_group = group_items[g]
            sorted_items = topological_sort_items_in_group(items_in_group)
            if len(sorted_items) != len(items_in_group):
                return []
            result.extend(sorted_items)
        
        return result
    
    def sortItems_dfs_approach(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        Approach 2: DFS-based Topological Sort
        
        Use DFS to perform topological sorting with cycle detection.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Assign groups to ungrouped items
        next_group_id = m
        for i in range(n):
            if group[i] == -1:
                group[i] = next_group_id
                next_group_id += 1
        
        # Build graphs
        group_graph = defaultdict(list)
        item_graph = defaultdict(list)
        
        group_items = defaultdict(list)
        for i in range(n):
            group_items[group[i]].append(i)
        
        # Build dependency graphs
        for i in range(n):
            for before_item in beforeItems[i]:
                item_graph[before_item].append(i)
                
                if group[before_item] != group[i]:
                    group_graph[group[before_item]].append(group[i])
        
        # DFS topological sort with cycle detection
        def dfs_topological_sort(graph, nodes):
            WHITE, GRAY, BLACK = 0, 1, 2
            color = {node: WHITE for node in nodes}
            result = []
            
            def dfs(node):
                if color[node] == GRAY:  # Cycle detected
                    return False
                if color[node] == BLACK:  # Already processed
                    return True
                
                color[node] = GRAY
                for neighbor in graph[node]:
                    if neighbor in color and not dfs(neighbor):
                        return False
                
                color[node] = BLACK
                result.append(node)
                return True
            
            for node in nodes:
                if color[node] == WHITE and not dfs(node):
                    return []
            
            return result[::-1]  # Reverse for topological order
        
        # Sort groups
        all_groups = list(group_items.keys())
        group_order = dfs_topological_sort(group_graph, all_groups)
        if not group_order:
            return []
        
        # Sort items within each group
        result = []
        for g in group_order:
            items = group_items[g]
            sorted_items = dfs_topological_sort(item_graph, items)
            if not sorted_items:
                return []
            result.extend(sorted_items)
        
        return result
    
    def sortItems_kahn_algorithm(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        """
        Approach 3: Kahn's Algorithm Implementation
        
        Use Kahn's algorithm for both group and item sorting.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Handle ungrouped items
        group_id = m
        for i in range(n):
            if group[i] == -1:
                group[i] = group_id
                group_id += 1
        
        # Build adjacency lists and indegree counts
        group_adj = defaultdict(set)
        group_indegree = defaultdict(int)
        
        item_adj = defaultdict(list)
        item_indegree = [0] * n
        
        # Group items
        group_to_items = defaultdict(list)
        for i in range(n):
            group_to_items[group[i]].append(i)
        
        # Initialize group indegrees
        for g in group_to_items:
            group_indegree[g] = 0
        
        # Process dependencies
        for curr in range(n):
            for prev in beforeItems[curr]:
                # Item level dependency
                item_adj[prev].append(curr)
                item_indegree[curr] += 1
                
                # Group level dependency
                prev_group, curr_group = group[prev], group[curr]
                if prev_group != curr_group and curr_group not in group_adj[prev_group]:
                    group_adj[prev_group].add(curr_group)
                    group_indegree[curr_group] += 1
        
        # Kahn's algorithm for groups
        def kahn_sort_groups():
            queue = deque()
            for g in group_indegree:
                if group_indegree[g] == 0:
                    queue.append(g)
            
            sorted_groups = []
            while queue:
                curr_group = queue.popleft()
                sorted_groups.append(curr_group)
                
                for next_group in group_adj[curr_group]:
                    group_indegree[next_group] -= 1
                    if group_indegree[next_group] == 0:
                        queue.append(next_group)
            
            return sorted_groups
        
        # Kahn's algorithm for items within a group
        def kahn_sort_items(items):
            # Build local graph
            local_adj = defaultdict(list)
            local_indegree = {item: 0 for item in items}
            
            for item in items:
                for next_item in item_adj[item]:
                    if next_item in local_indegree:
                        local_adj[item].append(next_item)
                        local_indegree[next_item] += 1
            
            # Kahn's algorithm
            queue = deque()
            for item in items:
                if local_indegree[item] == 0:
                    queue.append(item)
            
            sorted_items = []
            while queue:
                curr_item = queue.popleft()
                sorted_items.append(curr_item)
                
                for next_item in local_adj[curr_item]:
                    local_indegree[next_item] -= 1
                    if local_indegree[next_item] == 0:
                        queue.append(next_item)
            
            return sorted_items
        
        # Execute sorting
        sorted_groups = kahn_sort_groups()
        if len(sorted_groups) != len(group_to_items):
            return []
        
        result = []
        for g in sorted_groups:
            items = group_to_items[g]
            sorted_items = kahn_sort_items(items)
            if len(sorted_items) != len(items):
                return []
            result.extend(sorted_items)
        
        return result

def test_sort_items_by_groups():
    """Test sort items by groups algorithms"""
    solver = SortItemsByGroups()
    
    test_cases = [
        (8, 2, [-1,-1,1,0,0,1,0,-1], [[],[6],[5],[6],[3,6],[],[],[]], "Example 1"),
        (8, 2, [-1,-1,1,0,0,1,0,-1], [[],[6],[5],[6],[3],[],[4],[]], "Example 2 - no solution"),
        (5, 3, [0,0,2,1,1], [[1,3],[],[],[2],[]], "Simple case"),
    ]
    
    algorithms = [
        ("Dual Topological Sort", solver.sortItems_dual_topological_sort),
        ("DFS Approach", solver.sortItems_dfs_approach),
        ("Kahn's Algorithm", solver.sortItems_kahn_algorithm),
    ]
    
    print("=== Testing Sort Items by Groups Respecting Dependencies ===")
    
    for n, m, group, beforeItems, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"n={n}, m={m}, group={group}")
        print(f"beforeItems={beforeItems}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, m, group, beforeItems)
                print(f"{alg_name:20} | Result: {result}")
                
                # Validate result if not empty
                if result:
                    is_valid = len(result) == n and len(set(result)) == n
                    print(f"{'':20} | Valid: {is_valid}")
                
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:50]}")

if __name__ == "__main__":
    test_sort_items_by_groups()

"""
Sort Items by Groups Respecting Dependencies demonstrates
advanced topological sorting with hierarchical constraints,
combining group-level and item-level dependency resolution.
"""
