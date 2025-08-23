"""
2392. Build a Matrix With Conditions - Multiple Approaches
Difficulty: Hard

You are given a positive integer k. You are also given:
- a 2D integer array rowConditions of size n where rowConditions[i] = [abovei, belowi], and
- a 2D integer array colConditions of size m where colConditions[i] = [lefti, righti].

The two arrays contain integers from 1 to k.

You have to build a k x k matrix that contains each of the numbers from 1 to k exactly once. The remaining cells should have the value 0.

The matrix should also satisfy the following conditions:
- The number abovei should appear in a row with a smaller index than the number belowi for all i.
- The number lefti should appear in a column with a smaller index than the number righti for all i.

Return any matrix that satisfies the conditions. If no such matrix exists, return an empty matrix.
"""

from typing import List, Dict
from collections import defaultdict, deque

class BuildMatrixWithConditions:
    """Multiple approaches to build matrix with topological constraints"""
    
    def buildMatrix_dual_topological_sort(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        """
        Approach 1: Dual Topological Sort
        
        Perform topological sort for both row and column constraints.
        
        Time: O(k + E), Space: O(k²)
        """
        def topological_sort(conditions: List[List[int]]) -> List[int]:
            """Perform topological sort on given conditions"""
            graph = defaultdict(list)
            in_degree = defaultdict(int)
            
            # Initialize in-degree for all numbers
            for i in range(1, k + 1):
                in_degree[i] = 0
            
            # Build graph
            for before, after in conditions:
                graph[before].append(after)
                in_degree[after] += 1
            
            # Kahn's algorithm
            queue = deque()
            for i in range(1, k + 1):
                if in_degree[i] == 0:
                    queue.append(i)
            
            result = []
            while queue:
                node = queue.popleft()
                result.append(node)
                
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            # Check if all nodes are processed (no cycle)
            return result if len(result) == k else []
        
        # Get topological orders for rows and columns
        row_order = topological_sort(rowConditions)
        col_order = topological_sort(colConditions)
        
        # If either has a cycle, return empty matrix
        if not row_order or not col_order:
            return []
        
        # Create position mappings
        row_pos = {num: i for i, num in enumerate(row_order)}
        col_pos = {num: i for i, num in enumerate(col_order)}
        
        # Build the matrix
        matrix = [[0] * k for _ in range(k)]
        for num in range(1, k + 1):
            matrix[row_pos[num]][col_pos[num]] = num
        
        return matrix
    
    def buildMatrix_constraint_satisfaction(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        """
        Approach 2: Constraint Satisfaction Approach
        
        Model as constraint satisfaction problem.
        
        Time: O(k + E), Space: O(k²)
        """
        def has_cycle_and_get_order(conditions: List[List[int]]) -> tuple:
            """Check for cycle and return topological order"""
            graph = defaultdict(set)
            in_degree = {i: 0 for i in range(1, k + 1)}
            
            for u, v in conditions:
                if v not in graph[u]:
                    graph[u].add(v)
                    in_degree[v] += 1
            
            # Topological sort
            queue = deque()
            for i in range(1, k + 1):
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
            
            has_cycle = len(topo_order) != k
            return has_cycle, topo_order
        
        # Check both row and column constraints
        row_has_cycle, row_order = has_cycle_and_get_order(rowConditions)
        col_has_cycle, col_order = has_cycle_and_get_order(colConditions)
        
        if row_has_cycle or col_has_cycle:
            return []
        
        # Create position mappings
        row_position = {num: idx for idx, num in enumerate(row_order)}
        col_position = {num: idx for idx, num in enumerate(col_order)}
        
        # Construct matrix
        result = [[0] * k for _ in range(k)]
        for num in range(1, k + 1):
            row = row_position[num]
            col = col_position[num]
            result[row][col] = num
        
        return result
    
    def buildMatrix_dfs_cycle_detection(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        """
        Approach 3: DFS-based Cycle Detection
        
        Use DFS to detect cycles and find topological order.
        
        Time: O(k + E), Space: O(k²)
        """
        def dfs_topological_sort(conditions: List[List[int]]) -> List[int]:
            """DFS-based topological sort with cycle detection"""
            graph = defaultdict(list)
            for u, v in conditions:
                graph[u].append(v)
            
            WHITE, GRAY, BLACK = 0, 1, 2
            color = {i: WHITE for i in range(1, k + 1)}
            result = []
            
            def dfs(node: int) -> bool:
                if color[node] == GRAY:  # Back edge - cycle detected
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
            
            # Run DFS from all unvisited nodes
            for i in range(1, k + 1):
                if color[i] == WHITE:
                    if not dfs(i):
                        return []  # Cycle detected
            
            result.reverse()  # Reverse to get correct topological order
            return result
        
        # Get topological orders
        row_order = dfs_topological_sort(rowConditions)
        col_order = dfs_topological_sort(colConditions)
        
        if not row_order or not col_order:
            return []
        
        # Build matrix
        row_idx = {num: i for i, num in enumerate(row_order)}
        col_idx = {num: i for i, num in enumerate(col_order)}
        
        matrix = [[0] * k for _ in range(k)]
        for num in range(1, k + 1):
            matrix[row_idx[num]][col_idx[num]] = num
        
        return matrix
    
    def buildMatrix_optimized_kahn(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        """
        Approach 4: Optimized Kahn's Algorithm
        
        Optimized implementation of Kahn's algorithm.
        
        Time: O(k + E), Space: O(k²)
        """
        def optimized_topological_sort(edges: List[List[int]]) -> List[int]:
            """Optimized topological sort"""
            adj = [[] for _ in range(k + 1)]
            indegree = [0] * (k + 1)
            
            for u, v in edges:
                adj[u].append(v)
                indegree[v] += 1
            
            queue = deque()
            for i in range(1, k + 1):
                if indegree[i] == 0:
                    queue.append(i)
            
            order = []
            while queue:
                node = queue.popleft()
                order.append(node)
                
                for neighbor in adj[node]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)
            
            return order if len(order) == k else []
        
        # Get both orderings
        row_order = optimized_topological_sort(rowConditions)
        col_order = optimized_topological_sort(colConditions)
        
        if not row_order or not col_order:
            return []
        
        # Create result matrix
        row_map = {val: idx for idx, val in enumerate(row_order)}
        col_map = {val: idx for idx, val in enumerate(col_order)}
        
        result = [[0] * k for _ in range(k)]
        for num in range(1, k + 1):
            result[row_map[num]][col_map[num]] = num
        
        return result
    
    def buildMatrix_iterative_construction(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        """
        Approach 5: Iterative Matrix Construction
        
        Build matrix iteratively while checking constraints.
        
        Time: O(k + E), Space: O(k²)
        """
        def get_valid_ordering(constraints: List[List[int]]) -> List[int]:
            """Get valid ordering satisfying constraints"""
            graph = defaultdict(list)
            in_deg = defaultdict(int)
            
            # Initialize
            for i in range(1, k + 1):
                in_deg[i] = 0
            
            # Build constraint graph
            for before, after in constraints:
                graph[before].append(after)
                in_deg[after] += 1
            
            # Process nodes with no incoming edges
            available = deque()
            for i in range(1, k + 1):
                if in_deg[i] == 0:
                    available.append(i)
            
            ordering = []
            while available:
                current = available.popleft()
                ordering.append(current)
                
                # Update neighbors
                for next_node in graph[current]:
                    in_deg[next_node] -= 1
                    if in_deg[next_node] == 0:
                        available.append(next_node)
            
            # Validate complete ordering
            return ordering if len(ordering) == k else []
        
        # Get valid orderings for both dimensions
        row_ordering = get_valid_ordering(rowConditions)
        col_ordering = get_valid_ordering(colConditions)
        
        # Check if valid orderings exist
        if not row_ordering or not col_ordering:
            return []
        
        # Map numbers to their positions
        row_positions = {num: pos for pos, num in enumerate(row_ordering)}
        col_positions = {num: pos for pos, num in enumerate(col_ordering)}
        
        # Construct the final matrix
        matrix = [[0] * k for _ in range(k)]
        for number in range(1, k + 1):
            row = row_positions[number]
            col = col_positions[number]
            matrix[row][col] = number
        
        return matrix

def test_build_matrix_with_conditions():
    """Test build matrix with conditions algorithms"""
    solver = BuildMatrixWithConditions()
    
    test_cases = [
        (3, [[1,2],[3,2]], [[2,1],[3,2]], "Example 1"),
        (3, [[1,2],[2,3],[3,1]], [[2,1]], "Row cycle"),
        (2, [[1,2]], [[2,1]], "Valid 2x2"),
        (1, [], [], "Single element"),
    ]
    
    algorithms = [
        ("Dual Topological Sort", solver.buildMatrix_dual_topological_sort),
        ("Constraint Satisfaction", solver.buildMatrix_constraint_satisfaction),
        ("DFS Cycle Detection", solver.buildMatrix_dfs_cycle_detection),
        ("Optimized Kahn", solver.buildMatrix_optimized_kahn),
        ("Iterative Construction", solver.buildMatrix_iterative_construction),
    ]
    
    print("=== Testing Build Matrix With Conditions ===")
    
    for k, rowConditions, colConditions, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"k={k}, rowConditions={rowConditions}, colConditions={colConditions}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(k, rowConditions, colConditions)
                if result:
                    print(f"{alg_name:22} | ✓ | Matrix: {len(result)}x{len(result[0]) if result else 0}")
                    if k <= 3:  # Only print small matrices
                        for row in result:
                            print(f"{'':24}   {row}")
                else:
                    print(f"{alg_name:22} | ✓ | Empty matrix (no solution)")
            except Exception as e:
                print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_build_matrix_with_conditions()

"""
Build Matrix With Conditions demonstrates advanced topological
sorting applications for constraint satisfaction problems
with dual-dimensional ordering requirements.
"""
